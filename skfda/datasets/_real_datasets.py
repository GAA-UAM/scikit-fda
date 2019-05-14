import numpy as np
import rdata

from .. import FDataGrid
import warnings
from Cython.Build.Dependencies import package


def fdata_constructor(obj, attrs):
    names = obj["names"]

    return FDataGrid(data_matrix=obj["data"],
                     sample_points=obj["argvals"],
                     domain_range=obj["rangeval"],
                     dataset_label=names['main'][0],
                     axes_labels=[names['xlab'][0], names['ylab'][0]])


def functional_constructor(obj, attrs):
    name = obj['name']
    args_label = obj['args']
    values_label = obj['vals']
    target = np.array(obj['labels']).ravel()
    dataf = obj['dataf']

    sample_points_set = {a for o in dataf for a in o["args"]}

    args_init = min(sample_points_set)
    args_end = max(sample_points_set)

    sample_points = np.arange(args_init,
                              args_end + 1)

    data_matrix = np.zeros(shape=(len(dataf), len(sample_points)))

    for num_sample, o in enumerate(dataf):
        for t, x in zip(o["args"], o["vals"]):
            data_matrix[num_sample, t - args_init] = x

    return (FDataGrid(data_matrix=data_matrix,
                      sample_points=sample_points,
                      domain_range=(args_init, args_end),
                      dataset_label=name[0],
                      axes_labels=[args_label[0], values_label[0]]), target)


def fetch_cran(name, package_name, *, converter=None,
               **kwargs):
    """
    Fetch a dataset from CRAN.

    Args:
        name: Dataset name.
        package_name: Name of the R package containing the dataset.

    """
    import skdatasets

    if converter is None:
        converter = rdata.conversion.SimpleConverter({
            **rdata.conversion.DEFAULT_CLASS_MAP,
            "fdata": fdata_constructor,
            "functional": functional_constructor})

    return skdatasets.cran.fetch_dataset(name, package_name,
                                         converter=converter, **kwargs)


def fetch_ucr(name, **kwargs):
    """
    Fetch a dataset from the UCR.

    Args:
        name: Dataset name.

    Note:
        Functional multivariate datasets are not yet supported.

    References:
        Dau, Hoang Anh, Anthony Bagnall, Kaveh Kamgar, Chin-Chia Michael Yeh,
        Yan Zhu, Shaghayegh Gharghabi, Chotirat Ann Ratanamahatana, and
        Eamonn Keogh. «The UCR Time Series Archive».
        arXiv:1810.07758 [cs, stat], 17 de octubre de 2018.
        http://arxiv.org/abs/1810.07758.


    """
    import skdatasets

    dataset = skdatasets.ucr.fetch(name, **kwargs)

    def ucr_to_fdatagrid(data):
        if data.dtype == np.object_:
            data = np.array(data.tolist())

            # n_instances = data.shape[0]
            # dim_output = data.shape[1]
            # n_points = data.shape[2]

            data = np.transpose(data, axes=(0, 2, 1))

        sample_points = range(data.shape[1])

        return FDataGrid(data, sample_points=sample_points)

    dataset['data'] = ucr_to_fdatagrid(dataset['data'])
    del dataset['feature_names']

    data_test = dataset.get('data_test', None)
    if data_test is not None:
        dataset['data_test'] = ucr_to_fdatagrid(data_test)

    return dataset


def _fetch_cran_no_encoding_warning(*args, **kwargs):
    # Probably non thread safe
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning,
                                message="Unknown encoding. Assumed ASCII.")
        return fetch_cran(*args, **kwargs)


def _fetch_elem_stat_learn(name):
    return _fetch_cran_no_encoding_warning(name, "ElemStatLearn",
                                           version="0.1-7.1")


def _fetch_ddalpha(name):
    return _fetch_cran_no_encoding_warning(name, "ddalpha", version="1.3.4")


def _fetch_fda(name):
    return _fetch_cran_no_encoding_warning(name, "fda", version="2.4.7")


def _fetch_fda_usc(name):
    return _fetch_cran_no_encoding_warning(name, "fda.usc", version="1.3.0")


_param_descr = """
    Args:
        return_X_y: Return only the data and target as a tuple.
"""

_phoneme_descr = """
    These data arose from a collaboration between  Andreas Buja, Werner
    Stuetzle and Martin Maechler, and it is used as an illustration in the
    paper on Penalized Discriminant Analysis by Hastie, Buja and
    Tibshirani (1995).

    The data were extracted from the TIMIT database (TIMIT
    Acoustic-Phonetic Continuous Speech Corpus, NTIS, US Dept of Commerce)
    which is a widely used resource for research in speech recognition.  A
    dataset was formed by selecting five phonemes for
    classification based on digitized speech from this database.  The
    phonemes are transcribed as follows: "sh" as in "she", "dcl" as in
    "dark", "iy" as the vowel in "she", "aa" as the vowel in "dark", and
    "ao" as the first vowel in "water".  From continuous speech of 50 male
    speakers, 4509 speech frames of 32 msec duration were selected,
    approximately 2 examples of each phoneme from each speaker.  Each
    speech frame is represented by 512 samples at a 16kHz sampling rate,
    and each frame represents one of the above five phonemes.  The
    breakdown of the 4509 speech frames into phoneme frequencies is as
    follows:

    === ==== === ==== ===
     aa   ao dcl   iy  sh
    === ==== === ==== ===
    695 1022 757 1163 872
    === ==== === ==== ===

    From each speech frame, a log-periodogram was computed, which is one of
    several widely used methods for casting speech data in a form suitable
    for speech recognition.  Thus the data used in what follows consist of
    4509 log-periodograms of length 256, with known class (phoneme)
    memberships.

    The data contain curves sampled at 256 points, a response
    variable, and a column labelled "speaker" identifying the
    diffferent speakers.

    References:
        Hastie, Trevor; Buja, Andreas; Tibshirani, Robert. Penalized
        Discriminant Analysis. Ann. Statist. 23 (1995), no. 1, 73--102.
        doi:10.1214/aos/1176324456.
        https://projecteuclid.org/euclid.aos/1176324456
    """


def fetch_phoneme(return_X_y: bool = False):
    """
    Load the phoneme dataset.

    The data is obtained from the R package 'ElemStatLearn', which takes it
    from the dataset in `https://web.stanford.edu/~hastie/ElemStatLearn/`.

    """
    DESCR = _phoneme_descr

    raw_dataset = _fetch_elem_stat_learn("phoneme")

    data = raw_dataset["phoneme"]

    curve_data = data.iloc[:, 0:256]
    sound = data["g"].values
    speaker = data["speaker"].values

    curves = FDataGrid(data_matrix=curve_data.values,
                       sample_points=range(0, 256),
                       dataset_label="Phoneme",
                       axes_labels=["frequency", "log-periodogram"])

    if return_X_y:
        return curves, sound
    else:
        return {"data": curves,
                "target": sound.codes,
                "target_names": sound.categories.tolist(),
                "target_feature_names": ["sound"],
                "meta": np.array([speaker]).T,
                "meta_feature_names": ["speaker"],
                "DESCR": DESCR}


if hasattr(fetch_phoneme, "__doc__"):  # docstrings can be stripped off
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


def fetch_growth(return_X_y: bool = False):
    """
    Load the Berkeley Growth Study dataset.

    The data is obtained from the R package 'fda', which takes it from the
    Berkeley Growth Study.

    """
    DESCR = _growth_descr

    raw_dataset = _fetch_fda("growth")

    data = raw_dataset["growth"]

    ages = data["age"]
    females = data["hgtf"].T
    males = data["hgtm"].T

    curves = FDataGrid(data_matrix=np.concatenate((males, females), axis=0),
                       sample_points=ages,
                       dataset_label="Berkeley Growth Study",
                       axes_labels=["age", "height"])

    sex = np.array([0] * males.shape[0] + [1] * females.shape[0])

    if return_X_y:
        return curves, sex
    else:
        return {"data": curves,
                "target": sex,
                "target_names": ["male", "female"],
                "target_feature_names": ["sex"],
                "DESCR": DESCR}


if hasattr(fetch_growth, "__doc__"):  # docstrings can be stripped off
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


def fetch_tecator(return_X_y: bool = False):
    """
    Load the Tecator dataset.

    The data is obtained from the R package 'fda.usc', which takes it from
    http://lib.stat.cmu.edu/datasets/tecator.

    """
    DESCR = _tecator_descr

    raw_dataset = _fetch_fda_usc("tecator")

    data = raw_dataset["tecator"]

    curves = data['absorp.fdata']
    target = data['y'].values
    target_feature_names = data['y'].columns.values.tolist()

    if return_X_y:
        return curves, target
    else:
        return {"data": curves,
                "target": target,
                "target_feature_names": target_feature_names,
                "DESCR": DESCR}


if hasattr(fetch_tecator, "__doc__"):  # docstrings can be stripped off
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


def fetch_medflies(return_X_y: bool = False):
    """
    Load the Medflies dataset, where the flies are separated in two classes
    according to their longevity.

    The data is obtained from the R package 'ddalpha', which its a modification
    of the dataset in http://www.stat.ucdavis.edu/~wang/data/medfly1000.htm.

    """
    DESCR = _medflies_descr

    raw_dataset = _fetch_ddalpha("medflies")

    data = raw_dataset["medflies"]

    curves = data[0]

    unique = np.unique(data[1], return_inverse=True)
    target_names = [unique[0][1], unique[0][0]]
    target = 1 - unique[1]
    target_feature_names = ["lifetime"]

    if return_X_y:
        return curves, target
    else:
        return {"data": curves,
                "target": target,
                "target_names": target_names,
                "target_feature_names": target_feature_names,
                "DESCR": DESCR}


if hasattr(fetch_medflies, "__doc__"):  # docstrings can be stripped off
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


def fetch_weather(return_X_y: bool = False):
    """
    Load the Canadian Weather dataset.

    The data is obtained from the R package 'fda' from CRAN.

    """
    DESCR = _weather_descr

    raw_dataset = _fetch_fda("CanadianWeather")

    data = raw_dataset["CanadianWeather"]

    weather_daily = np.asarray(data["dailyAv"])

    # Axes 0 and 1 must be transposed since in the downloaded dataset the
    # data_matrix shape is (nfeatures, nsamples, ndim_image) while our
    # data_matrix shape is (nsamples, nfeatures, ndim_image).
    temp_prec_daily = np.transpose(weather_daily[:, :, 0:2], axes=(1, 0, 2))

    curves = FDataGrid(data_matrix=temp_prec_daily,
                       sample_points=range(1, 366),
                       dataset_label="Canadian Weather",
                       axes_labels=["day", "temperature (ºC)",
                                    "precipitation (mm.)"])

    target_names, target = np.unique(data["region"], return_inverse=True)

    if return_X_y:
        return curves, target
    else:
        return {"data": curves,
                "target": target,
                "target_names": target_names,
                "target_feature_names": ["region"],
                "DESCR": DESCR}


if hasattr(fetch_weather, "__doc__"):  # docstrings can be stripped off
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


def fetch_aemet(return_X_y: bool = False):
    """
    Load the Spanish Weather dataset.

    The data is obtained from the R package 'fda.usc' from CRAN.

    """
    DESCR = _aemet_descr

    raw_dataset = _fetch_fda_usc("aemet")

    data = raw_dataset["aemet"]

    fd_temp = data["temp"]
    fd_logprec = data["logprec"]
    fd_wind = data["wind.speed"]

    if return_X_y:
        return fd_temp, fd_logprec, fd_wind
    else:
        return {"data": (fd_temp, fd_logprec, fd_wind),
                "meta": np.asarray(data["df"])[:,
                                               np.array([0, 1, 2, 3, 6, 7])],
                "meta_names": ["ind", "name", "province", "altitude",
                               "longitude", "latitude"],
                "meta_feature_names": ["location"],
                "DESCR": DESCR}


if hasattr(fetch_aemet, "__doc__"):  # docstrings can be stripped off
    fetch_aemet.__doc__ += _aemet_descr + _param_descr
