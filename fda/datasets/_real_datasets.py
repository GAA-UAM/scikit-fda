
import textwrap

import numpy as np
import rdata
import skdatasets

from ..grid import FDataGrid


def fdata_constructor(obj, attrs):
    names = obj["names"]

    return FDataGrid(data_matrix=obj["data"],
                     sample_points=obj["argvals"],
                     sample_range=obj["rangeval"],
                     dataset_label=names['main'][0],
                     axes_labels=[names['xlab'][0], names['ylab'][0]])


def fetch_cran_dataset(dataset_name, package_name, *, converter=None,
                       **kwargs):
    """
    Fetch a dataset from CRAN.

    """
    if converter is None:
        converter = rdata.conversion.SimpleConverter({
            **rdata.conversion.DEFAULT_CLASS_MAP,
            "fdata": fdata_constructor})

    return skdatasets.cran.fetch_dataset(dataset_name, package_name,
                                         converter=converter, **kwargs)


_phoneme_descr = textwrap.dedent("""
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

      aa   ao dcl   iy  sh
     695 1022 757 1163 872

    From each speech frame, a log-periodogram was computed, which is one of
    several widely used methods for casting speech data in a form suitable
    for speech recognition.  Thus the data used in what follows consist of
    4509 log-periodograms of length 256, with known class (phoneme)
    memberships.

    The data contain curves sampled at 256 points, a response
    variable, and a column labelled "speaker" identifying the
    diffferent speakers.

    Hastie, Trevor; Buja, Andreas; Tibshirani, Robert. Penalized Discriminant
    Analysis. Ann. Statist. 23 (1995), no. 1, 73--102.
    doi:10.1214/aos/1176324456. https://projecteuclid.org/euclid.aos/1176324456
    """)


def fetch_phoneme(return_X_y=False):
    """Load the phoneme dataset.

    The data is obtained from the R package 'ElemStatLearn', which takes it
    from the dataset in `https://web.stanford.edu/~hastie/ElemStatLearn/`.

    """
    DESCR = _phoneme_descr

    raw_dataset = fetch_cran_dataset(
        "phoneme.RData", "ElemStatLearn",
        package_url="https://cran.r-project.org/src/contrib/Archive/ElemStatLearn/ElemStatLearn_0.1-7.1.tar.gz")

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
    fetch_phoneme.__doc__ += _phoneme_descr

_growth_descr = textwrap.dedent("""
    The Berkeley Growth Study (Tuddenham and Snyder, 1954) recorded the
    heights of 54 girls and 39 boys between the ages of 1 and 18 years.
    Heights were measured at 31 ages for each child, and the standard
    error of these measurements was about 3mm, tending to be larger in
    early childhood and lower in later years.

    Tuddenham, R. D., and Snyder, M. M. (1954) "Physical growth of California
    boys and girls from birth to age 18", University of California
    Publications in Child Development, 1, 183-364.
""")


def fetch_growth(return_X_y=False):
    """Load the Berkeley Growth Study dataset.

    The data is obtained from the R package 'fda', which takes it from the
    Berkeley Growth Study.

    """
    DESCR = _growth_descr

    raw_dataset = fetch_cran_dataset(
        "growth.rda", "fda",
        package_url="https://cran.r-project.org/src/contrib/Archive/fda/fda_2.4.7.tar.gz")

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
    fetch_growth.__doc__ += _growth_descr

_tecator_descr = textwrap.dedent("""
    This is the Tecator data set: The task is to predict the fat content of a
    meat sample on the basis of its near infrared absorbance spectrum.
    -------------------------------------------------------------------------
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
""")


def fetch_tecator(return_X_y=False):
    """Load the Tecator dataset.

    The data is obtained from the R package 'fda.usc', which takes it from
    http://lib.stat.cmu.edu/datasets/tecator.

    """
    DESCR = _tecator_descr

    raw_dataset = fetch_cran_dataset(
        "tecator.rda", "fda.usc",
        package_url="https://cran.r-project.org/src/contrib/Archive/fda.usc/fda.usc_1.3.0.tar.gz")

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
    fetch_tecator.__doc__ += _tecator_descr
