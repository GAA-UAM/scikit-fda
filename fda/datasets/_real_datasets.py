
import textwrap

import numpy as np
import rdata
import skdatasets

from ..grid import FDataGrid


def fdata_constructor(obj, attrs):
    names_tuple = tuple(n[1] for n in obj["names"])

    return FDataGrid(data_matrix=obj["data"],
                     sample_points=obj["argvals"],
                     sample_range=obj["rangeval"],
                     dataset_label=names_tuple[0],
                     axes_labels=names_tuple[1:])


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


def load_phoneme(return_X_y=False):
    """Load the phoneme dataset.

    The data is obtained from the R package 'ElemStatLearn', which takes it
    from the dataset in `https://web.stanford.edu/~hastie/ElemStatLearn/`.

    """
    DESCR = textwrap.dedent("""
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
    """)

    raw_dataset = fetch_cran_dataset(
        "phoneme.RData", "ElemStatLearn",
        package_url="https://cran.r-project.org/src/contrib/Archive/ElemStatLearn/ElemStatLearn_0.1-7.1.tar.gz")

    data = raw_dataset["phoneme"]

    curve_data = data.iloc[:, 0:256]
    sound = data["g"].values
    speaker = data["speaker"].values

    curves = FDataGrid(data_matrix=curve_data.values,
                       sample_points=range(0, 256))

    if return_X_y:
        return curves, sound
    else:
        return {"data": curves,
                "target": sound,
                "meta": np.array([speaker]).T,
                "meta_names": "speaker",
                "DESCR": DESCR}
