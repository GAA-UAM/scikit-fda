"""Module for functional data manipulation.

Defines the abstract class that should be implemented by the funtional data
objects of the package and contains some commons methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas
from typing_extensions import override

from .._utils.ndfunction import NDFunction, concatenate as concatenate
from ..typing._base import LabelTuple, LabelTupleLike
from .extrapolation import ExtrapolationLike


class FData(  # noqa: WPS214
    ABC,
    NDFunction,
    pandas.api.extensions.ExtensionArray,  # type: ignore[misc]
):
    """
    Defines the structure of a functional data object.

    Attributes:
        n_samples: Number of samples.
        dim_domain: Dimension of the domain.
        dim_codomain: Dimension of the image.
        extrapolation: Default extrapolation mode.
        dataset_name: Name of the dataset.
        argument_names: Tuple containing the names of the different
            arguments.
        coordinate_names: Tuple containing the names of the different
            coordinate functions.

    """

    dataset_name: str | None

    def __init__(
        self,
        *,
        extrapolation: ExtrapolationLike | None = None,
        dataset_name: str | None = None,
        argument_names: LabelTupleLike | None = None,
        coordinate_names: LabelTupleLike | None = None,
        sample_names: LabelTupleLike | None = None,
    ) -> None:

        self.extrapolation = extrapolation  # type: ignore[assignment]
        self.dataset_name = dataset_name

        self.argument_names = argument_names  # type: ignore[assignment]
        self.coordinate_names = coordinate_names  # type: ignore[assignment]
        self.sample_names = sample_names  # type: ignore[assignment]

    @property
    def argument_names(self) -> LabelTuple:
        return self._argument_names

    @argument_names.setter
    def argument_names(
        self,
        names: LabelTupleLike | None,
    ) -> None:
        if names is None:
            names = (None,) * self.dim_domain
        else:
            names = tuple(names)
            if len(names) != self.dim_domain:
                raise ValueError(
                    "There must be a name for each of the "
                    "dimensions of the domain.",
                )

        self._argument_names = names

    @property
    def coordinate_names(self) -> LabelTuple:
        return self._coordinate_names

    @coordinate_names.setter
    def coordinate_names(
        self,
        names: LabelTupleLike | None,
    ) -> None:
        if names is None:
            names = (None,) * self.dim_codomain
        else:
            names = tuple(names)
            if len(names) != self.dim_codomain:
                raise ValueError(
                    "There must be a name for each of the "
                    "dimensions of the codomain.",
                )

        self._coordinate_names = names

    @property
    def sample_names(self) -> LabelTuple:
        return self._sample_names

    @sample_names.setter
    def sample_names(self, names: LabelTupleLike | None) -> None:
        if names is None:
            names = (None,) * self.n_samples
        else:
            names = tuple(names)
            if len(names) != self.n_samples:
                raise ValueError(
                    "There must be a name for each of the samples.",
                )

        self._sample_names = names

    @property
    @abstractmethod
    def n_samples(self) -> int:
        """Return the number of samples.

        Returns:
            Number of samples of the FData object.

        """
        pass

    @override
    @property
    def shape(self) -> tuple[int, ...]:
        return (self.n_samples,)

    @property
    @abstractmethod
    def dim_domain(self) -> int:
        """Return number of dimensions of the :term:`domain`.

        Returns:
            Number of dimensions of the domain.

        """
        pass

    @override
    @property
    def input_shape(self) -> tuple[int, ...]:
        return (self.dim_domain,)

    @property
    @abstractmethod
    def dim_codomain(self) -> int:
        """Return number of dimensions of the :term:`codomain`.

        Returns:
            Number of dimensions of the codomain.

        """
        pass

    @override
    @property
    def output_shape(self) -> tuple[int, ...]:
        return (self.dim_codomain,)
