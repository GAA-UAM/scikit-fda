from __future__ import annotations

import warnings
from typing import Any, Mapping, Tuple, overload

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.utils import Bunch
from typing_extensions import Literal

import rdata

from ..representation import FDataGrid
from ..typing._numpy import NDArrayFloat, NDArrayInt


def _get_skdatasets_repositories() -> Any:
    import skdatasets

    repositories = getattr(skdatasets, "repositories", None)
    if repositories is None:
        repositories = skdatasets

    return repositories


def fdata_constructor(
    obj: Any,
    attrs: Mapping[str | bytes, Any],
) -> FDataGrid:
    """
    Construct a :func:`FDataGrid` objet from a R `fdata` object.

    This constructor can be used in the dict passed to
    :func:`rdata.conversion.SimpleConverter` in order to
    convert `fdata` objects from the fda.usc package.

    """
    names = obj["names"]

    return FDataGrid(
        data_matrix=obj["data"],
        grid_points=obj["argvals"],
        domain_range=obj["rangeval"],
        dataset_name=names['main'][0],
        argument_names=(names['xlab'][0],),
        coordinate_names=(names['ylab'][0],),
    )


def functional_constructor(
    obj: Any,
    attrs: Mapping[str | bytes, Any],
) -> Tuple[FDataGrid, NDArrayInt]:
    """
    Construct a :func:`FDataGrid` objet from a R `functional` object.

    This constructor can be used in the dict passed to
    :func:`rdata.conversion.SimpleConverter` in order to
    convert `functional` objects from the ddalpha package.

    """
    name = obj['name']
    args_label = obj['args']
    values_label = obj['vals']
    target = np.array(obj['labels']).ravel()
    dataf = obj['dataf']

    grid_points_set = {a for o in dataf for a in o["args"]}

    args_init = min(grid_points_set)
    args_end = max(grid_points_set)

    grid_points = np.arange(args_init, args_end + 1)

    data_matrix = np.zeros(shape=(len(dataf), len(grid_points)))

    for num_sample, o in enumerate(dataf):
        for t, x in zip(o["args"], o["vals"]):
            data_matrix[num_sample, t - args_init] = x

    return (
        FDataGrid(
            data_matrix=data_matrix,
            grid_points=grid_points,
            domain_range=(args_init, args_end),
            dataset_name=name[0],
            argument_names=(args_label[0],),
            coordinate_names=(values_label[0],),
        ),
        target,
    )


def fetch_cran(
    name: str,
    package_name: str,
    *,
    converter: rdata.conversion.Converter | None = None,
    **kwargs: Any,
) -> Any:
    """
    Fetch a dataset from CRAN.

    Args:
        name: Dataset name.
        package_name: Name of the R package containing the dataset.
        converter: Object that performs the conversion of the R objects to
            Python objects.
        kwargs: Additional parameters for the function
            :func:`skdatasets.repositories.cran.fetch_dataset`.

    Returns:
        The dataset, with the R types converted to suitable Python
        types.

    """
    repositories = _get_skdatasets_repositories()

    if converter is None:
        converter = rdata.conversion.SimpleConverter({
            **rdata.conversion.DEFAULT_CLASS_MAP,
            "fdata": fdata_constructor,
            "functional": functional_constructor,
        })

    return repositories.cran.fetch_dataset(
        name,
        package_name,
        converter=converter,
        **kwargs,
    )


def _ucr_to_fdatagrid(name: str, data: NDArrayFloat) -> FDataGrid:
    if data.dtype == np.object_:
        data = np.array(data.tolist())

        # n_instances := data.shape[0]
        # dim_output  := data.shape[1]
        # n_points    := data.shape[2]

        data = np.transpose(data, axes=(0, 2, 1))

    grid_points = range(data.shape[1])

    return FDataGrid(data, grid_points=grid_points, dataset_name=name)


@overload
def fetch_ucr(
    name: str,
    *,
    return_X_y: Literal[False] = False,
    **kwargs: Any,
) -> Bunch:
    pass


@overload
def fetch_ucr(
    name: str,
    *,
    return_X_y: Literal[True],
    **kwargs: Any,
) -> Tuple[FDataGrid, NDArrayInt]:
    pass


def fetch_ucr(
    name: str,
    *,
    return_X_y: bool = False,
    **kwargs: Any,
) -> Bunch | Tuple[FDataGrid, NDArrayInt]:
    """
    Fetch a dataset from the UCR.

    Args:
        name: Dataset name.
        kwargs: Additional parameters for the function
            :func:`skdatasets.repositories.ucr.fetch`.

    Returns:
        The dataset requested.

    Note:
        Functional multivariate datasets are not yet supported.

    References:
        Dau, Hoang Anh, Anthony Bagnall, Kaveh Kamgar, Chin-Chia Michael Yeh,
        Yan Zhu, Shaghayegh Gharghabi, Chotirat Ann Ratanamahatana, and
        Eamonn Keogh. «The UCR Time Series Archive».
        arXiv:1810.07758 [cs, stat], 17 de octubre de 2018.
        http://arxiv.org/abs/1810.07758.


    """
    repositories = _get_skdatasets_repositories()

    dataset = repositories.ucr.fetch(name, **kwargs)

    dataset['data'] = _ucr_to_fdatagrid(
        name=dataset['name'],
        data=dataset['data'],
    )
    dataset.pop('feature_names')

    if return_X_y:
        return dataset['data'], dataset['target']

    return dataset


def _fetch_cran_no_encoding_warning(*args: Any, **kwargs: Any) -> Any:
    # Probably non thread safe
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="Unknown encoding. Assumed ASCII.",
        )
        return fetch_cran(*args, **kwargs)


def _fetch_elem_stat_learn(name: str) -> Any:
    return _fetch_cran_no_encoding_warning(
        name,
        "ElemStatLearn",
        version="0.1-7.1",
    )


def _fetch_ddalpha(name: str) -> Any:
    return _fetch_cran_no_encoding_warning(name, "ddalpha", version="1.3.4")


def _fetch_fda(name: str) -> Any:
    return _fetch_cran_no_encoding_warning(name, "fda", version="2.4.7")


def _fetch_fda_usc(name: str) -> Any:
    return _fetch_cran_no_encoding_warning(name, "fda.usc", version="1.3.0")


_param_descr = """
    Args:
        return_X_y: Return only the data and target as a tuple.
        as_frame: Return the data in a Pandas Dataframe or Series.
"""

_phoneme_descr = """
    These data arose from a collaboration between  Andreas Buja, Werner
    Stuetzle and Martin Maechler, and it is used as an illustration in the
    paper on Penalized Discriminant Analysis by Hastie, Buja and
    Tibshirani (1995).

    The data were extracted from the TIMIT database (TIMIT
    Acoustic-Phonetic Continuous Speech Corpus, NTIS, US Dept of Commerce)
    which is a widely used resource for research in speech recognition. A
    dataset was formed by selecting five phonemes for
    classification based on digitized speech from this database.   
    phonemes are transcribed as follows: "sh" as in "she", "dcl" as in
    "dark", "iy" as the vowel in "she", "aa" as the vowel in "dark", and
    "ao" as the first vowel in "water". From continuous speech of 50 male
    speakers, 4509 speech frames of 32 msec duration were selected,
    approximately 2 examples of each phoneme from each speaker.  Each
    speech frame is represented by 512 samples at a 16kHz sampling rate,
    and each frame represents one of the above five phonemes. The
    breakdown of the 4509 speech frames into phoneme frequencies is as
    follows:

    === ==== === ==== ===
     aa   ao dcl   iy  sh
    === ==== === ==== ===
    695 1022 757 1163 872
    === ==== === ==== ===

    From each speech frame, a log-periodogram was computed, which is one of
    several widely used methods for casting speech data in a form suitable
    for speech recognition. Thus the data used in what follows consist of
    4509 log-periodograms of length 256, with known class (phoneme)
    memberships.

    The data contain curves sampled at 256 points, a response
    variable, and a column labelled "speaker" identifying the
    different speakers.

    References:
        Hastie, Trevor; Buja, Andreas; Tibshirani, Robert. Penalized
        Discriminant Analysis. Ann. Statist. 23 (1995), no. 1, 73--102.
        doi:10.1214/aos/1176324456.
        https://projecteuclid.org/euclid.aos/1176324456
    """


@overload
def fetch_phoneme(
    *,
    return_X_y: Literal[False] = False,
    as_frame: bool = False,
) -> Bunch:
    pass


@overload
def fetch_phoneme(
    *,
    return_X_y: Literal[True],
    as_frame: Literal[False] = False,
) -> Tuple[FDataGrid, NDArrayInt]:
    pass


@overload
def fetch_phoneme(
    *,
    return_X_y: Literal[True],
    as_frame: Literal[True],
) -> Tuple[DataFrame, Series]:
    pass


def fetch_phoneme(
    *,
    return_X_y: bool = False,
    as_frame: bool = False,
) -> Bunch | Tuple[FDataGrid, NDArrayInt] | Tuple[DataFrame, Series]:
    """
    Load the phoneme dataset.

    The data is obtained from the R package 'ElemStatLearn', which takes it
    from the dataset in `https://web.stanford.edu/~hastie/ElemStatLearn/`.

    """
    descr = _phoneme_descr

    raw_dataset = _fetch_elem_stat_learn("phoneme")

    data = raw_dataset["phoneme"]

    n_points = 256

    curve_data = data.iloc[:, 0:n_points]
    sound = data["g"].values
    speaker = data["speaker"].values

    curves = FDataGrid(
        data_matrix=curve_data.values,
        grid_points=np.linspace(0, 8, n_points),
        domain_range=[0, 8],
        dataset_name="Phoneme",
        argument_names=("frequency (kHz)",),
        coordinate_names=("log-periodogram",),
    )

    curve_name = "log-periodogram"
    target_name = "phoneme"
    frame = None

    if as_frame:
        frame = pd.DataFrame({
            curve_name: curves,
            target_name: sound,
        })
        curves = frame.iloc[:, [0]]
        target = frame.iloc[:, 1]
        meta = pd.Series(speaker, name="speaker")
    else:
        target = sound.codes
        meta = np.array([speaker]).T

    if return_X_y:
        return curves, target

    return Bunch(
        data=curves,
        target=target,
        frame=frame,
        categories={target_name: sound.categories.tolist()},
        feature_names=[curve_name],
        target_names=[target_name],
        meta=meta,
        meta_names=["speaker"],
        DESCR=descr,
    )


if fetch_phoneme.__doc__ is not None:  # docstrings can be stripped off
    fetch_phoneme.__doc__ += _phoneme_descr + _param_descr

_growth_descr = """
    The Berkeley Growth Study (Tuddenham and Snyder, 1954) recorded the
    heights of 54 girls and 39 boys between the ages of 1 and 18 years.
    Heights were measured at 31 ages for each child, and the standard
    error of these measurements was about 3mm, tending to be larger in
    early childhood and lower in later years.

    References:
        Tuddenham, R. D., and Snyder, M. M. (1954) "Physical growth of
        California boys and girls from birth to age 18",
        University of California Publications in Child Development,
        1, 183-364.
"""


@overload
def fetch_growth(
    *,
    return_X_y: Literal[False] = False,
    as_frame: bool = False,
) -> Bunch:
    pass


@overload
def fetch_growth(
    *,
    return_X_y: Literal[True],
    as_frame: Literal[False] = False,
) -> Tuple[FDataGrid, NDArrayInt]:
    pass


@overload
def fetch_growth(
    *,
    return_X_y: Literal[True],
    as_frame: Literal[True],
) -> Tuple[DataFrame, Series]:
    pass


def fetch_growth(
    return_X_y: bool = False,
    as_frame: bool = False,
) -> Bunch | Tuple[FDataGrid, NDArrayInt] | Tuple[DataFrame, Series]:
    """
    Load the Berkeley Growth Study dataset.

    The data is obtained from the R package 'fda', which takes it from the
    Berkeley Growth Study.

    """
    descr = _growth_descr

    raw_dataset = _fetch_fda("growth")

    data = raw_dataset["growth"]

    ages = data["age"]
    females = data["hgtf"].T
    males = data["hgtm"].T

    sex = np.array([0] * males.shape[0] + [1] * females.shape[0])
    curves = FDataGrid(
        data_matrix=np.concatenate((males, females), axis=0),
        grid_points=ages,
        dataset_name="Berkeley Growth Study",
        argument_names=("age",),
        coordinate_names=("height",),
    )

    curve_name = "height"
    target_name = "sex"
    target_categories = ["male", "female"]
    frame = None

    if as_frame:
        sex = pd.Categorical.from_codes(sex, categories=target_categories)
        frame = pd.DataFrame({
            curve_name: curves,
            target_name: sex,
        })
        curves = frame.iloc[:, [0]]
        sex = frame.iloc[:, 1]

    if return_X_y:
        return curves, sex

    return Bunch(
        data=curves,
        target=sex,
        frame=frame,
        categories={target_name: target_categories},
        feature_names=[curve_name],
        target_names=[target_name],
        DESCR=descr,
    )


if fetch_growth.__doc__ is not None:  # docstrings can be stripped off
    fetch_growth.__doc__ += _growth_descr + _param_descr

_tecator_descr = """
    This is the Tecator data set: The task is to predict the fat content of a
    meat sample on the basis of its near infrared absorbance spectrum.

    1. Statement of permission from Tecator (the original data source)

    These data are recorded on a Tecator Infratec Food and Feed Analyzer
    working in the wavelength range 850 - 1050 nm by the Near Infrared
    Transmission (NIT) principle. Each sample contains finely chopped pure
    meat with different moisture, fat and protein contents.

    If results from these data are used in a publication we want you to
    mention the instrument and company name (Tecator) in the publication.
    In addition, please send a preprint of your article to

        Karin Thente, Tecator AB,
        Box 70, S-263 21 Hoganas, Sweden

    The data are available in the public domain with no responsability from
    the original data source. The data can be redistributed as long as this
    permission note is attached.

    For more information about the instrument - call Perstorp Analytical's
    representative in your area.


    2. Description of the data

    For each meat sample the data consists of a 100 channel spectrum of
    absorbances and the contents of moisture (water), fat and protein.
    The absorbance is -log10 of the transmittance
    measured by the spectrometer. The three contents, measured in percent,
    are determined by analytic chemistry.

    There are 215 samples.
"""


@overload
def fetch_tecator(
    *,
    return_X_y: Literal[False] = False,
    as_frame: bool = False,
) -> Bunch:
    pass


@overload
def fetch_tecator(
    *,
    return_X_y: Literal[True],
    as_frame: Literal[False] = False,
) -> Tuple[FDataGrid, NDArrayFloat]:
    pass


@overload
def fetch_tecator(
    *,
    return_X_y: Literal[True],
    as_frame: Literal[True],
) -> Tuple[DataFrame, DataFrame]:
    pass


def fetch_tecator(
    return_X_y: bool = False,
    as_frame: bool = False,
) -> Bunch | Tuple[FDataGrid, NDArrayFloat] | Tuple[DataFrame, DataFrame]:
    """
    Load the Tecator dataset.

    The data is obtained from the R package 'fda.usc', which takes it from
    http://lib.stat.cmu.edu/datasets/tecator.

    """
    descr = _tecator_descr

    raw_dataset = _fetch_fda_usc("tecator")

    data = raw_dataset["tecator"]

    curves = data['absorp.fdata']
    target = data['y'].rename(columns=str.lower)
    feature_name = curves.dataset_name.lower()
    target_names = target.columns.values.tolist()

    frame = None

    if as_frame:
        curves = pd.DataFrame({feature_name: curves})
        frame = pd.concat([curves, target], axis=1)
    else:
        target = target.values

    if return_X_y:
        return curves, target

    return Bunch(
        data=curves,
        target=target,
        frame=frame,
        categories={},
        feature_names=[feature_name],
        target_names=target_names,
        DESCR=descr,
    )


if fetch_tecator.__doc__ is not None:  # docstrings can be stripped off
    fetch_tecator.__doc__ += _tecator_descr + _param_descr

_medflies_descr = """
    The data set medfly1000.dat  consists of number of eggs laid daily for
    each of 1000 medflies (Mediterranean fruit flies, Ceratitis capitata)
    until time of death. Data were obtained in Dr. Carey's laboratory.
    A description of the experiment which was done by Professor Carey of
    UC Davis and collaborators in a medfly rearing facility in
    Mexico is in Carey et al.(1998) below. The main questions are to
    explore the relationship of age patterns of fecundity to mortality,
    longevity and lifetime reproduction.

    A basic finding was that individual mortality is associated with the
    time-dynamics of the egg-laying trajectory. An approximate parametric model
    of the egg laying process was developed and used in Müller et al. (2001).
    Nonparametric approaches which extend principal component analysis for
    curve data to the situation when covariates are present have been
    developed and discussed in  Chiou, Müller and Wang (2003)
    and Chiou et al. (2003).

    References:
        Carey, J.R., Liedo, P., Müller, H.G., Wang, J.L., Chiou, J.M. (1998).
        Relationship of age patterns of fecundity to mortality, longevity,
        and lifetime reproduction in a large cohort of Mediterranean fruit
        fly females. J. of Gerontology --Biological Sciences 53, 245-251.

        Chiou, J.M., Müller, H.G., Wang, J.L. (2003). Functional
        quasi-likelihood regression models with smooth random effects.
        J. Royal Statist. Soc. B65, 405-423. (PDF)

        Chiou, J.M., Müller, H.G., Wang, J.L., Carey, J.R. (2003). A functional
        multiplicative effects model for longitudinal data, with
        application to reproductive histories of female medflies.
        Statistica Sinica 13, 1119-1133. (PDF)

        Chiou, J.M., Müller, H.G., Wang, J.L. (2004).Functional response
        models. Statistica Sinica 14,675-693. (PDF)
"""


@overload
def fetch_medflies(
    *,
    return_X_y: Literal[False] = False,
    as_frame: bool = False,
) -> Bunch:
    pass


@overload
def fetch_medflies(
    *,
    return_X_y: Literal[True],
    as_frame: Literal[False] = False,
) -> Tuple[FDataGrid, NDArrayInt]:
    pass


@overload
def fetch_medflies(
    *,
    return_X_y: Literal[True],
    as_frame: Literal[True],
) -> Tuple[DataFrame, Series]:
    pass


def fetch_medflies(
    return_X_y: bool = False,
    as_frame: bool = False,
) -> Bunch | Tuple[FDataGrid, NDArrayInt] | Tuple[DataFrame, Series]:
    """
    Load the Medflies dataset.

    The data is obtained from the R package 'ddalpha', which its a modification
    of the dataset in http://www.stat.ucdavis.edu/~wang/data/medfly1000.htm.

    """
    descr = _medflies_descr

    raw_dataset = _fetch_ddalpha("medflies")

    data = raw_dataset["medflies"]

    curves = data[0]

    unique = np.unique(data[1], return_inverse=True)
    target_categories = [unique[0][1], unique[0][0]]
    target = 1 - unique[1]
    curve_name = 'eggs'
    target_name = "lifetime"

    frame = None

    if as_frame:
        target = pd.Categorical.from_codes(
            target,
            categories=target_categories,
        )
        frame = pd.DataFrame({
            curve_name: curves,
            target_name: target,
        })
        curves = frame.iloc[:, [0]]
        target = frame.iloc[:, 1]

    if return_X_y:
        return curves, target

    return Bunch(
        data=curves,
        target=target,
        frame=frame,
        categories={target_name: target_categories},
        feature_names=[curve_name],
        target_names=[target_name],
        DESCR=descr,
    )


if fetch_medflies.__doc__ is not None:  # docstrings can be stripped off
    fetch_medflies.__doc__ += _medflies_descr + _param_descr

_weather_descr = """
    Daily temperature and precipitation at 35 different locations in Canada
    averaged over 1960 to 1994.

    References:
        Ramsay, James O., and Silverman, Bernard W. (2006),
        Functional Data Analysis, 2nd ed. , Springer, New York.

        Ramsay, James O., and Silverman, Bernard W. (2002),
        Applied Functional Data Analysis, Springer, New York
"""


@overload
def fetch_weather(
    *,
    return_X_y: Literal[False] = False,
    as_frame: bool = False,
) -> Bunch:
    pass


@overload
def fetch_weather(
    *,
    return_X_y: Literal[True],
    as_frame: Literal[False] = False,
) -> Tuple[FDataGrid, NDArrayInt]:
    pass


@overload
def fetch_weather(
    *,
    return_X_y: Literal[True],
    as_frame: Literal[True],
) -> Tuple[DataFrame, Series]:
    pass


def fetch_weather(
    return_X_y: bool = False,
    as_frame: bool = False,
) -> Bunch | Tuple[FDataGrid, NDArrayInt] | Tuple[DataFrame, Series]:
    """
    Load the Canadian Weather dataset.

    The data is obtained from the R package 'fda' from CRAN.

    """
    descr = _weather_descr

    data = _fetch_fda("CanadianWeather")["CanadianWeather"]

    # Axes 0 and 1 must be transposed since in the downloaded dataset the
    # data_matrix shape is (nfeatures, n_samples, dim_codomain) while our
    # data_matrix shape is (n_samples, nfeatures, dim_codomain).
    temp_prec_daily = np.transpose(
        np.asarray(data["dailyAv"])[:, :, 0:2], axes=(1, 0, 2),
    )

    days_in_year = 365

    curves = FDataGrid(
        data_matrix=temp_prec_daily,
        grid_points=np.arange(0, days_in_year) + 0.5,
        domain_range=(0, days_in_year),
        dataset_name="Canadian Weather",
        sample_names=data["place"],
        argument_names=("day",),
        coordinate_names=(
            "temperature (ºC)",
            "precipitation (mm.)",
        ),
    )

    curve_name = "daily averages"
    target_name = "region"
    target_categories, target = np.unique(data["region"], return_inverse=True)

    frame = None

    if as_frame:
        target = pd.Categorical.from_codes(
            target,
            categories=target_categories,
        )
        frame = pd.DataFrame({
            curve_name: curves,
            "place": data["place"],
            "province": data["province"],
            "latitude": np.asarray(data["coordinates"])[:, 0],
            "longitude": np.asarray(data["coordinates"])[:, 1],
            "index": data["geogindex"],
            "monthly temperatures": np.asarray(
                data["monthlyTemp"],
            ).T.tolist(),
            "monthly precipitation": np.asarray(
                data["monthlyPrecip"],
            ).T.tolist(),
            target_name: target,
        })
        X = frame.iloc[:, :-1]
        target = frame.iloc[:, -1]
        feature_names = list(X.columns.values)

        additional_dict = {}
    else:
        feature_names = [curve_name]
        X = curves
        meta = np.concatenate(
            (
                np.array(data["place"], dtype=np.object_)[:, np.newaxis],
                np.array(data["province"], dtype=np.object_)[:, np.newaxis],
                np.asarray(data["coordinates"], dtype=np.object_),
                np.array(data["geogindex"], dtype=np.object_)[:, np.newaxis],
                np.asarray(data["monthlyTemp"]).T.tolist(),
                np.asarray(data["monthlyPrecip"]).T.tolist(),
            ),
            axis=1,
        )
        meta_names = [
            "place",
            "province",
            "latitude",
            "longitude",
            "index",
            "monthly temperatures",
            "monthly precipitation",
        ]

        additional_dict = {
            "meta": meta,
            "meta_names": meta_names,
        }

    if return_X_y:
        return X, target

    return Bunch(
        data=X,
        target=target,
        frame=frame,
        categories={target_name: target_categories},
        feature_names=feature_names,
        target_names=[target_name],
        **additional_dict,
        DESCR=descr,
    )


if fetch_weather.__doc__ is not None:  # docstrings can be stripped off
    fetch_weather.__doc__ += _weather_descr + _param_descr

_aemet_descr = """
    Series of daily summaries of 73 spanish weather stations selected for the
    period 1980-2009. The dataset contains the geographic information of each
    station and the average for the period 1980-2009 of daily temperature,
    daily precipitation and daily wind speed. Meteorological State Agency of
    Spain (AEMET), http://www.aemet.es/. Government of Spain.

    Authors:
        Manuel Febrero Bande, Manuel Oviedo de la Fuente <manuel.oviedo@usc.es>

    Source:
        The data were obtained from the FTP of AEMET in 2009.
"""


@overload
def fetch_aemet(
    *,
    return_X_y: Literal[False] = False,
    as_frame: bool = False,
) -> Bunch:
    pass


@overload
def fetch_aemet(
    *,
    return_X_y: Literal[True],
    as_frame: Literal[False] = False,
) -> Tuple[FDataGrid, None]:
    pass


@overload
def fetch_aemet(
    *,
    return_X_y: Literal[True],
    as_frame: Literal[True],
) -> Tuple[DataFrame, None]:
    pass


def fetch_aemet(  # noqa: WPS210
    return_X_y: bool = False,
    as_frame: bool = False,
) -> Bunch | Tuple[FDataGrid, None] | Tuple[DataFrame, None]:
    """
    Load the Spanish Weather dataset.

    The data is obtained from the R package 'fda.usc' from CRAN.

    """
    descr = _aemet_descr

    data = _fetch_fda_usc("aemet")["aemet"]

    days_in_year = 365

    data_matrix = np.empty((73, days_in_year, 3))
    data_matrix[:, :, 0] = data["temp"].data_matrix[:, :, 0]
    data_matrix[:, :, 1] = data["logprec"].data_matrix[:, :, 0]
    data_matrix[:, :, 2] = data["wind.speed"].data_matrix[:, :, 0]

    curves = data["temp"].copy(
        data_matrix=data_matrix,
        grid_points=np.arange(0, days_in_year) + 0.5,
        domain_range=(0, days_in_year),
        dataset_name="AEMET",
        sample_names=data["df"].iloc[:, 1],
        argument_names=("day",),
        coordinate_names=(
            "temperature (ºC)",
            "logprecipitation",
            "wind speed (m/s)",
        ),
    )

    curve_name = "daily averages"
    df_names = [
        "index",
        "place",
        "province",
        "altitude",
        "longitude",
        "latitude",
    ]
    df_indexes = np.array([0, 1, 2, 3, 6, 7])

    frame = None

    if as_frame:
        frame = pd.DataFrame({
            curve_name: curves,
            **{
                n: data["df"].iloc[:, d]
                for (n, d) in zip(df_names, df_indexes)
            },
        })
        X = frame
        feature_names = list(X.columns.values)

        additional_dict = {}

    else:
        feature_names = [curve_name]
        X = curves
        meta = np.asarray(data["df"])[:, df_indexes]
        meta_names = df_names
        additional_dict = {
            "meta": meta,
            "meta_names": meta_names,
        }

    if return_X_y:
        return X, None

    return Bunch(
        data=X,
        target=None,
        frame=frame,
        categories={},
        feature_names=feature_names,
        **additional_dict,
        DESCR=descr,
    )


if fetch_aemet.__doc__ is not None:  # docstrings can be stripped off
    fetch_aemet.__doc__ += _aemet_descr + _param_descr


_octane_descr = """
    Near infrared (NIR) spectra of gasoline samples, with wavelengths ranging
    from 1102nm to 1552nm with measurements every two nm.
    This dataset contains six outliers to which ethanol was added, which is
    required in some states. See [RDEH2006]_ and [HuRS2015]_ for further
    details.

    The data is labeled according to this different composition.

    Source:
        Esbensen K. (2001). Multivariate data analysis in practice. 5th edn.
        Camo Software, Trondheim, Norway.

    References:
        ..  [RDEH2006] Rousseeuw, Peter & Debruyne, Michiel & Engelen, Sanne &
            Hubert, Mia. (2006). Robustness and Outlier Detection in
            Chemometrics. Critical Reviews in Analytical Chemistry. 36.
            221-242. 10.1080/10408340600969403.
        ..  [HuRS2015] Hubert, Mia & Rousseeuw, Peter & Segaert, Pieter.
            (2015). Multivariate functional outlier detection. Statistical
            Methods and Applications. 24. 177-202. 10.1007/s10260-015-0297-8.

"""


@overload
def fetch_octane(
    *,
    return_X_y: Literal[False] = False,
    as_frame: bool = False,
) -> Bunch:
    pass


@overload
def fetch_octane(
    *,
    return_X_y: Literal[True],
    as_frame: Literal[False] = False,
) -> Tuple[FDataGrid, NDArrayInt]:
    pass


@overload
def fetch_octane(
    *,
    return_X_y: Literal[True],
    as_frame: Literal[True],
) -> Tuple[DataFrame, Series]:
    pass


def fetch_octane(
    return_X_y: bool = False,
    as_frame: bool = False,
) -> Bunch | Tuple[FDataGrid, NDArrayInt] | Tuple[DataFrame, Series]:
    """Load near infrared spectra of gasoline samples.

    This function fetchs the octane dataset from the R package 'mrfDepth'
    from CRAN.

    """
    descr = _octane_descr

    # octane file from mrfDepth R package
    raw_dataset = fetch_cran("octane", "mrfDepth", version="1.0.11")
    data = raw_dataset['octane'][..., 0].T

    # The R package only stores the values of the curves, but the paper
    # describes the rest of the data. According to [RDEH2006], Section 5.4:

    # "wavelengths ranging from 1102nm to 1552nm with measurements every two
    # nm.""
    wavelength_start = 1102
    wavelength_end = 1552
    wavelength_count = 226

    grid_points = np.linspace(
        wavelength_start,
        wavelength_end,
        wavelength_count,
    )

    # "The octane data set contains six outliers (25, 26, 36–39) to which
    # alcohol was added".
    target = np.zeros(len(data), dtype=np.bool_)
    target[24:26] = 1  # noqa: WPS432
    target[35:39] = 1  # noqa: WPS432

    target_name = "is outlier"

    curve_name = "absorbances"

    curves = FDataGrid(
        data,
        grid_points=grid_points,
        dataset_name="octane",
        argument_names=("wavelength (nm)",),
        coordinate_names=("absorbances",),
    )

    frame = None

    if as_frame:
        frame = pd.DataFrame({
            curve_name: curves,
            target_name: target,
        })
        curves = frame.iloc[:, [0]]
        target = frame.iloc[:, 1]

    if return_X_y:
        return curves, target

    return Bunch(
        data=curves,
        target=target,
        frame=frame,
        categories={},
        feature_names=[curve_name],
        target_names=[target_name],
        DESCR=descr,
    )


if fetch_octane.__doc__ is not None:  # docstrings can be stripped off
    fetch_octane.__doc__ += _octane_descr + _param_descr

_gait_descr = """
    Angles formed by the hip and knee of each of 39 children over each boy
    gait cycle.

    References:
        Ramsay, James O., and Silverman, Bernard W. (2006),
        Functional Data Analysis, 2nd ed. , Springer, New York.

        Ramsay, James O., and Silverman, Bernard W. (2002),
        Applied Functional Data Analysis, Springer, New York
"""


@overload
def fetch_gait(
    *,
    return_X_y: Literal[False] = False,
    as_frame: bool = False,
) -> Bunch:
    pass


@overload
def fetch_gait(
    *,
    return_X_y: Literal[True],
    as_frame: Literal[False] = False,
) -> Tuple[FDataGrid, None]:
    pass


@overload
def fetch_gait(
    *,
    return_X_y: Literal[True],
    as_frame: Literal[True],
) -> Tuple[DataFrame, None]:
    pass


def fetch_gait(
    return_X_y: bool = False,
    as_frame: bool = False,
) -> Bunch | Tuple[FDataGrid, None] | Tuple[DataFrame, None]:
    """
    Load the GAIT dataset.

    The data is obtained from the R package 'fda' from CRAN.

    """
    descr = _gait_descr

    raw_data = _fetch_fda("gait")

    data = raw_data["gait"]

    data_matrix = np.asarray(data)
    data_matrix = np.transpose(data_matrix, axes=(1, 0, 2))
    grid_points = np.asarray(data.coords.get('dim_0'), dtype=np.float64)
    sample_names = list(
        np.asarray(data.coords.get('dim_1'), dtype=np.str_),
    )
    feature_name = 'gait'

    curves = FDataGrid(
        data_matrix=data_matrix,
        grid_points=grid_points,
        dataset_name=feature_name,
        sample_names=sample_names,
        argument_names=("time (proportion of gait cycle)",),
        coordinate_names=(
            "hip angle (degrees)",
            "knee angle (degrees)",
        ),
    )

    frame = None

    if as_frame:
        curves = pd.DataFrame({feature_name: curves})
        frame = curves

    if return_X_y:
        return curves, None

    return Bunch(
        data=curves,
        target=None,
        frame=frame,
        categories={},
        feature_names=[feature_name],
        target_names=[],
        DESCR=descr,
    )


if fetch_gait.__doc__ is not None:  # docstrings can be stripped off
    fetch_gait.__doc__ += _gait_descr + _param_descr

_handwriting_descr = """
    Data representing the X-Y coordinates along time obtained while
    writing the word "fda". The sample contains 20 instances measured over
    2.3 seconds that had been aligned for a better understanding. Each instance
    is formed by 1401 coordinate values.

    References:
        Ramsay, James O., and Silverman, Bernard W. (2006),
        Functional Data Analysis, 2nd ed. , Springer, New York.
"""


@overload
def fetch_handwriting(
    *,
    return_X_y: Literal[False] = False,
    as_frame: bool = False,
) -> Bunch:
    pass


@overload
def fetch_handwriting(
    *,
    return_X_y: Literal[True],
    as_frame: Literal[False] = False,
) -> Tuple[FDataGrid, None]:
    pass


@overload
def fetch_handwriting(
    *,
    return_X_y: Literal[True],
    as_frame: Literal[True],
) -> Tuple[DataFrame, None]:
    pass


def fetch_handwriting(
    return_X_y: bool = False,
    as_frame: bool = False,
) -> Bunch | Tuple[FDataGrid, None] | Tuple[DataFrame, None]:
    """
    Load the HANDWRIT dataset.

    The data is obtained from the R package 'fda' from CRAN.

    """
    descr = _handwriting_descr

    raw_data = _fetch_fda("handwrit")

    data = raw_data["handwrit"]

    data_matrix = np.asarray(data)
    data_matrix = np.transpose(data_matrix, axes=(1, 0, 2))
    grid_points = np.asarray(data.coords.get('dim_0'), dtype=np.float64)
    sample_names = list(
        np.asarray(data.coords.get('dim_1'), dtype=np.str_),
    )
    feature_name = 'handwrit'

    curves = FDataGrid(
        data_matrix=data_matrix,
        grid_points=grid_points,
        dataset_name=feature_name,
        sample_names=sample_names,
        argument_names=("time",),
        coordinate_names=(
            "x coordinates",
            "y coordinates",
        ),
    )

    frame = None

    if as_frame:
        curves = pd.DataFrame({feature_name: curves})
        frame = curves

    if return_X_y:
        return curves, None

    return Bunch(
        data=curves,
        target=None,
        frame=frame,
        categories={},
        feature_names=[feature_name],
        target_names=[],
        DESCR=descr,
    )


if fetch_handwriting.__doc__ is not None:  # docstrings can be stripped off
    fetch_handwriting.__doc__ += _handwriting_descr + _param_descr

_nox_descr_template = """
    NOx levels measured every hour by a control station in Poblenou in
    Barcelona (Spain) {cite}.

    References:
        {bibliography}

"""

_nox_descr = _nox_descr_template.format(
    cite="[1]",
    bibliography="[1] M. Febrero, P. Galeano, and W. González‐Manteiga, "
    "“Outlier detection in functional data by depth measures, with "
    "application to identify abnormal NOx levels,” Environmetrics, vol. 19, "
    "no. 4, pp. 331–345, Jun. 2008, doi: 10.1002/env.878.",
)


@overload
def fetch_nox(
    *,
    return_X_y: Literal[False] = False,
    as_frame: bool = False,
) -> Bunch:
    pass


@overload
def fetch_nox(
    *,
    return_X_y: Literal[True],
    as_frame: Literal[False] = False,
) -> Tuple[FDataGrid, NDArrayInt]:
    pass


@overload
def fetch_nox(
    *,
    return_X_y: Literal[True],
    as_frame: Literal[True],
) -> Tuple[DataFrame, DataFrame]:
    pass


def fetch_nox(
    return_X_y: bool = False,
    as_frame: bool = False,
) -> Bunch | Tuple[FDataGrid, NDArrayInt] | Tuple[DataFrame, DataFrame]:
    """
    Load the NOx dataset.

    The data is obtained from the R package 'fda.usc'.

    """
    descr = _nox_descr

    raw_dataset = _fetch_fda_usc("poblenou")

    data = raw_dataset["poblenou"]

    curves = data['nox']
    target = data['df'].iloc[:, 2]
    weekend = (
        (data['df'].iloc[:, 1] == "6")
        | (data['df'].iloc[:, 1] == "7")
    )
    target[weekend] = "1"
    target = pd.Series(
        target.values.codes.astype(np.bool_),
        name="festive day",
    )
    curves.coordinate_names = ["$mglm^3$"]
    feature_name = curves.dataset_name.lower()
    target_names = target.values.tolist()

    frame = None

    if as_frame:
        curves = pd.DataFrame({feature_name: curves})
        frame = pd.concat([curves, target], axis=1)
    else:
        target = target.values

    if return_X_y:
        return curves, target

    return Bunch(
        data=curves,
        target=target,
        frame=frame,
        categories={},
        feature_names=[feature_name],
        target_names=target_names,
        DESCR=descr,
    )


if fetch_nox.__doc__ is not None:  # docstrings can be stripped off
    fetch_nox.__doc__ += _nox_descr_template.format(
        cite=":footcite:`febrero++_2008_outlier`",
        bibliography=".. footbibliography::"
    ) + _param_descr

_mco_descr_template = """
    The mithochondiral calcium overload (MCO) was measured in two groups
    (control and treatment) every 10 seconds during an hour in isolated mouse
    cardiac cells. In fact, due to technical reasons, the original experiment
    [see {cite}] was performed twice, using both the
    "intact", original cells and "permeabilized" cells (a condition related
    to the mitochondrial membrane).

    References:
        {bibliography}

"""

_mco_descr = _mco_descr_template.format(
    cite="Ruiz-Meana et. al. (2003)",
    bibliography="[1] M. Ruiz-Meana, D. Garcia-Dorado, P. Pina, J. Inserte, "
    "L. Agulló, and J. Soler-Soler, “Cariporide preserves mitochondrial "
    "proton gradient and delays ATP depletion in cardiomyocytes during "
    "ischemic conditions,” Am. J. Physiol. Heart Circ. Physiol., vol. 285, "
    "no. 3, pp. H999-1006, Sep. 2003, doi: 10.1152/ajpheart.00035.2003.",
)


@overload
def fetch_mco(
    *,
    return_X_y: Literal[False] = False,
    as_frame: bool = False,
) -> Bunch:
    pass


@overload
def fetch_mco(
    *,
    return_X_y: Literal[True],
    as_frame: Literal[False] = False,
) -> Tuple[FDataGrid, NDArrayInt]:
    pass


@overload
def fetch_mco(
    *,
    return_X_y: Literal[True],
    as_frame: Literal[True],
) -> Tuple[DataFrame, DataFrame]:
    pass


def fetch_mco(
    return_X_y: bool = False,
    as_frame: bool = False,
) -> Bunch | Tuple[FDataGrid, NDArrayInt] | Tuple[DataFrame, DataFrame]:
    """
    Load the mithochondiral calcium overload (MCO) dataset.

    The data is obtained from the R package 'fda.usc'.

    """
    descr = _mco_descr

    raw_dataset = _fetch_fda_usc("MCO")

    data = raw_dataset["MCO"]

    curves = data['intact']
    target = pd.Series(
        data['classintact'].rename_categories(["control", "treatment"]),
        name="group",
    )
    feature_name = curves.dataset_name.lower()
    target_names = target.values.tolist()

    frame = None

    if as_frame:
        curves = pd.DataFrame({feature_name: curves})
        frame = pd.concat([curves, target], axis=1)
    else:
        target = target.values.codes

    if return_X_y:
        return curves, target

    return Bunch(
        data=curves,
        target=target,
        frame=frame,
        categories={},
        feature_names=[feature_name],
        target_names=target_names,
        DESCR=descr,
    )


if fetch_mco.__doc__ is not None:  # docstrings can be stripped off
    fetch_mco.__doc__ += _mco_descr_template.format(
        cite=":footcite:`ruiz++_2003_cariporide`",
        bibliography=".. footbibliography::",
    ) + _param_descr
