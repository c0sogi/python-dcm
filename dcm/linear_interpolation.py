from typing import NamedTuple, Self

import numpy as np
import numpy.typing as npt


class Locator1D(NamedTuple):
    xL: npt.NDArray[np.signedinteger]
    xR: npt.NDArray[np.signedinteger]
    wL: npt.NDArray[np.floating]
    wR: npt.NDArray[np.floating]

    @classmethod
    def compute(
        cls,
        X: np.ndarray,
        x: np.ndarray,
    ) -> Self:
        """
        Computes the indices and interpolation weights for 1D linear interpolation.

        Parameters:
        - X: 1D array, strictly increasing grid points.
        - x: 1D array, query points for interpolation.

        Returns:
        Locator1D tuple containing:
        - xL: Indices of the grid points to the left or equal to the query points.
        - xR: Indices of the grid points to the right or equal to the query points.
        - wL: Interpolation weights for the left grid points.
        - wR: Interpolation weights for the right grid points.
        """
        size: int = X.size
        X = X.astype(np.float64)
        x = x.astype(np.float64)

        xR: np.ndarray = np.clip(
            np.searchsorted(X, x, side="right") - 1, 0, size - 1
        )
        xL: np.ndarray = np.clip(
            np.searchsorted(X, x, side="left"), 0, size - 1
        )

        denom = X[xR] - X[xL]
        wR: np.ndarray = np.empty_like(x)
        mask: np.ndarray = denom != 0
        wR[mask] = np.clip((x[mask] - X[xL[mask]]) / denom[mask], 0, 1)
        wR[~mask] = 0.5

        return cls(xL=xL, xR=xR, wL=1.0 - wR, wR=wR)


class Locator2D(NamedTuple):
    yBL: npt.NDArray[np.signedinteger]
    yBR: npt.NDArray[np.signedinteger]
    yTL: npt.NDArray[np.signedinteger]
    yTR: npt.NDArray[np.signedinteger]
    xBL: npt.NDArray[np.signedinteger]
    xBR: npt.NDArray[np.signedinteger]
    xTL: npt.NDArray[np.signedinteger]
    xTR: npt.NDArray[np.signedinteger]
    wBL: npt.NDArray[np.floating]
    wBR: npt.NDArray[np.floating]
    wTL: npt.NDArray[np.floating]
    wTR: npt.NDArray[np.floating]

    @classmethod
    def compute(
        cls,
        X: npt.NDArray[np.floating],
        Y: npt.NDArray[np.floating],
        x: npt.NDArray[np.floating],
        y: npt.NDArray[np.floating],
    ) -> Self:
        """
        Computes the indices and interpolation weights for 2D bilinear interpolation.

        Parameters:
        - X, Y: 1D arrays of strictly increasing grid points for the x and y axes.
        - x, y: 1D arrays of query points where interpolation is desired.

        Returns:
        Locator2D tuple containing:
        - xBL, xBR, xTL, xTR: Indices of the grid points surrounding the query point in the x direction.
        - yBL, yBR, yTL, yTR: Indices of the grid points surrounding the query point in the y direction.
        - wBL, wBR, wTL, wTR: Weights for each surrounding grid point (bottom-left, bottom-right, top-left, top-right).
        """
        size_x: int = X.size
        size_y: int = Y.size

        X = X.astype(np.float64)
        Y = Y.astype(np.float64)
        x = x.astype(np.float64)
        y = y.astype(np.float64)

        # x-axis interpolation indices
        idx_right = np.searchsorted(X, x, side="right") - 1
        idx_right = np.clip(idx_right, 0, size_x - 2)
        idx_left = idx_right + 1

        denom_x = X[idx_left] - X[idx_right]
        xfrac = np.empty_like(x)
        mask_x = denom_x != 0
        xfrac[mask_x] = np.clip(
            (x[mask_x] - X[idx_right[mask_x]]) / denom_x[mask_x], 0, 1
        )
        xfrac[~mask_x] = 0.5  # If equal x values, average.

        # y-axis interpolation indices
        idy_right = np.searchsorted(Y, y, side="right") - 1
        idy_right = np.clip(idy_right, 0, size_y - 2)
        idy_left = idy_right + 1

        denom_y = Y[idy_left] - Y[idy_right]
        yfrac = np.empty_like(y)
        mask_y = denom_y != 0
        yfrac[mask_y] = np.clip(
            (y[mask_y] - Y[idy_right[mask_y]]) / denom_y[mask_y], 0, 1
        )
        yfrac[~mask_y] = 0.5  # If equal y values, average.

        # Compute weights for the four surrounding points
        wBL = (1 - xfrac) * (1 - yfrac)  # Bottom-left
        wBR = xfrac * (1 - yfrac)  # Bottom-right
        wTL = (1 - xfrac) * yfrac  # Top-left
        wTR = xfrac * yfrac  # Top-right

        # Compute the four surrounding indices
        yBL = idy_right  # Bottom-left y index
        yBR = idy_right  # Bottom-right y index
        yTL = idy_left  # Top-left y index
        yTR = idy_left  # Top-right y index

        xBL = idx_right  # Bottom-left x index
        xBR = idx_left  # Bottom-right x index
        xTL = idx_right  # Top-left x index
        xTR = idx_left  # Top-right x index

        return cls(
            yBL=yBL,
            yBR=yBR,
            yTL=yTL,
            yTR=yTR,
            xBL=xBL,
            xBR=xBR,
            xTL=xTL,
            xTR=xTR,
            wBL=wBL,
            wBR=wBR,
            wTL=wTL,
            wTR=wTR,
        )


def lininterp1(
    X: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating],
    x: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """
    Linear interpolation, using pre-computed contributions for efficiency.

    Parameters:
    - X: 1D array of strictly increasing grid points.
    - V: 1D array of values at the grid points.
    - x: 1D array of query points for interpolation.

    Returns:
    1D array of interpolated values at the query points.
    """
    loc = Locator1D.compute(X, x)
    return V[loc.xR] * loc.wR + V[loc.xL] * loc.wL


def lininterp2(
    X: npt.NDArray[np.floating],
    Y: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating],
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """
    Perform 2D bilinear interpolation.

    Parameters:
    - X, Y: 1D arrays of strictly increasing grid points for the x and y axes.
    - V: 2D array of values at the grid points (shape: (len(Y), len(X))).
    - x, y: 1D arrays of query points where interpolation is desired.

    Returns:
    1D array of interpolated values at the query points.
    """

    # Compute the interpolation contributions
    loc = Locator2D.compute(X, Y, x, y)

    # Extract values at the four surrounding points
    vBL = V[loc.yBL, loc.xBL]  # Bottom-left
    vBR = V[loc.yBR, loc.xBR]  # Bottom-right
    vTL = V[loc.yTL, loc.xTL]  # Top-left
    vTR = V[loc.yTR, loc.xTR]  # Top-right

    # Compute the interpolated value
    return vBL * loc.wBL + vBR * loc.wBR + vTL * loc.wTL + vTR * loc.wTR
