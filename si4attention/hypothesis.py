"""Module for hypothesis."""

import torch

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(mode=True)

torch.manual_seed(0)


class Hypothesis:
    """An abstract class for hypothesis."""

    def __init__(self) -> None:
        """Initialize the class."""
        self.name: str
        self.tau: torch.Tensor
        self.sign_vector: torch.Tensor

    def construct_hypothesis(
        self,
        cov: torch.Tensor,
        attention_map: torch.Tensor,
        data: torch.Tensor,
        reference: torch.Tensor | None = None,
    ) -> tuple[float, torch.Tensor, torch.Tensor]:
        """Construct the hypothesis based on the observed data.

        Parameters
        ----------
        cov: torch.Tensor
            The covariance matrix or scalar specifying the variance.
        attention_map: torch.Tensor
            The attention map.
        data: torch.Tensor
            The observed data.
        reference: torch.Tensor | None, optional
            The observed reference data if any. Defaults to None.

        Returns
        -------
        tuple[float, torch.Tensor, torch.Tensor]
            The observed test statistic, and the vectors specifying the search space.
        """
        stat, data, sigma_eta, eta_sigma_eta = self._process(
            cov,
            attention_map,
            data,
            reference,
            is_observed=True,
        )
        b = sigma_eta / torch.sqrt(eta_sigma_eta)
        a = data.flatten() - stat * b
        return stat.detach().item(), a, b

    def compute_test_statistic(
        self,
        cov: torch.Tensor,
        attention_map: torch.Tensor,
        data: torch.Tensor,
        reference: torch.Tensor | None = None,
        *,
        is_observed: bool = False,
    ) -> torch.Tensor:
        """Compute the test statistic.

        Parameters
        ----------
        cov: torch.Tensor
            The covariance matrix or scalar specifying the variance.
        attention_map: torch.Tensor
            The attention map.
        data: torch.Tensor
            The input data.
        reference: torch.Tensor | None, optional
            The reference data if any. Defaults to None.
        is_observed: bool, optional
            Whether the data is observed or not. Defaults to False.

        Returns
        -------
        float
            The test statistic.
        """
        stat, _, _, _ = self._process(
            cov,
            attention_map,
            data,
            reference,
            is_observed=is_observed,
        )
        return stat

    def _process(
        self,
        cov: torch.Tensor,
        attention_map: torch.Tensor,
        data: torch.Tensor,
        reference: torch.Tensor | None,
        *,
        is_observed: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the test statistic with additional information.

        Parameters
        ----------
        cov: torch.Tensor
            The covariance matrix or scalar specifying the variance.
        attention_map: torch.Tensor
            The attention map.
        data: torch.Tensor
            The input data.
        reference: torch.Tensor | None
            The reference data if any. Defaults to None.
        is_observed: bool
            Whether the data is observed or not. Defaults to False.

        Returns
        -------
        tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]
            The test statistic, the data concatenated with the reference data if any,
            the sigma eta product vector, and the eta sigma eta product.
        """
        eta = self.compute_eta_vector(attention_map, is_observed=is_observed)
        if reference is not None:
            data = torch.cat([data, reference])
            if len(cov.shape) > 0:
                cov = torch.block_diag(cov, cov)
        if len(cov.shape) == 0:
            sigma_eta = cov * eta
            eta_sigma_eta = cov * torch.linalg.norm(eta) ** 2
        else:
            sigma_eta = cov @ eta
            eta_sigma_eta = eta @ sigma_eta
        stat = eta @ data.flatten() / torch.sqrt(eta_sigma_eta)
        return stat, data, sigma_eta, eta_sigma_eta

    def compute_eta_vector(
        self,
        attention_map: torch.Tensor,
        *,
        is_observed: bool,
    ) -> torch.Tensor:
        """Compute the eta vector.

        Parameters
        ----------
        attention_map: torch.Tensor
            The attention map.
        is_observed: bool
            Whether the data is observed or not.

        Returns
        -------
        torch.Tensor
            The eta vector.
        """
        raise NotImplementedError

    def convert_to_value(self, attention_map: torch.Tensor) -> torch.Tensor:
        """Convert the attention map to the function value.

        Parameters
        ----------
        attention_map: torch.Tensor
            The attention map.

        Returns
        -------
        torch.Tensor
            The function value.
        """
        data_dim = self.sign_vector.numel()
        return (self.tau - attention_map.reshape((-1, data_dim))) * self.sign_vector

    def convert_to_derivative(self, attention_map_jvp: torch.Tensor) -> torch.Tensor:
        """Convert the jacobian vector product of the attention map to the derivative of the function.

        Parameters
        ----------
        attention_map_jvp: torch.Tensor
            Jacobian vector product of the attention map with respect to
            the search direction.

        Returns
        -------
        torch.Tensor
            The derivative of the function.
        """
        data_dim = self.sign_vector.numel()
        return -attention_map_jvp.reshape((-1, data_dim)) * self.sign_vector


class Background(Hypothesis):
    """Class for hypothesis to compare with the background."""

    def __init__(self, tau: float = 0.6) -> None:
        """Initialize the hypothesis to compare with the background."""
        self.name = "background"
        self.tau = torch.tensor(tau).double()

    def compute_eta_vector(
        self,
        attention_map: torch.Tensor,
        *,
        is_observed: bool,
    ) -> torch.Tensor:
        """Compute the eta vector to compare with the background.

        Parameters
        ----------
        attention_map: torch.Tensor
            The attention map.
        is_observed: bool
            Whether the data is observed or not.

        Returns
        -------
        torch.Tensor
            The eta vector.
        """
        region = (attention_map > self.tau).flatten()
        if is_observed:
            self.sign_vector = torch.where(region, 1.0, -1.0).double()
        back = torch.logical_not(region)
        region, back = region.double(), back.double()
        return region / region.sum() - back / back.sum()


class Neighbor(Hypothesis):
    """Class for hypothesis to compare with the neighbor."""

    def __init__(
        self,
        data_type: str,
        data_size: int,
        tau: float = 0.6,
        radius: int | None = None,
    ) -> None:
        """Initialize the hypothesis to compare with the neighbor."""
        self.name = "neighbor"
        self.data_type = data_type
        match data_type:
            case "image":
                self.radius = max(1, data_size // 16)
            case "series":
                self.radius = data_size // 16
        if radius is not None:
            self.radius = radius
        self.tau = torch.tensor(tau).double()

    def compute_eta_vector(
        self,
        attention_map: torch.Tensor,
        *,
        is_observed: bool,
    ) -> torch.Tensor:
        """Compute the eta vector to compare with the neighbor.

        Parameters
        ----------
        attention_map: torch.Tensor
            The attention map.
        is_observed: bool
            Whether the data is observed or not.

        Returns
        -------
        torch.Tensor
            The eta vector.
        """
        region = (attention_map > self.tau).unsqueeze(1)
        if is_observed:
            self.sign_vector = torch.where(region.flatten(), 1.0, -1.0).double()
        if self.data_type == "image":
            pooled = torch.nn.functional.max_pool2d(
                region.float(),
                kernel_size=1 + 2 * self.radius,
                stride=1,
                padding=self.radius,
            )
        else:
            pooled = torch.nn.functional.max_pool1d(
                region.float(),
                kernel_size=1 + 2 * self.radius,
                stride=1,
                padding=self.radius,
            )
        neighbor = torch.logical_xor(region, pooled.bool())
        region, neighbor = region.flatten().double(), neighbor.flatten().double()
        return region / region.sum() - neighbor / neighbor.sum()


class Reference(Hypothesis):
    """Class for hypothesis to compare with the reference."""

    def __init__(self, tau: float = 0.6) -> None:
        """Initialize the hypothesis to compare with the reference."""
        self.name = "reference"
        self.tau = torch.tensor(tau).double()

    def compute_eta_vector(
        self,
        attention_map: torch.Tensor,
        *,
        is_observed: bool,
    ) -> torch.Tensor:
        """Compute the eta vector to compare with the reference.

        Parameters
        ----------
        attention_map: torch.Tensor
            The attention map.
        is_observed: bool
            Whether the data is observed or not.

        Returns
        -------
        torch.Tensor
            The eta vector.
        """
        region = (attention_map > self.tau).flatten()
        if is_observed:
            self.sign_vector = torch.where(region, 1.0, -1.0).double()
        region = region.double()
        return torch.cat([region, -region]) / region.sum()
