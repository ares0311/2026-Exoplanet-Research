"""Root conftest.py — pre-load astroquery into sys.modules before any test
modifies sys.path.  Without this, Skills tests that insert their own parent
directory into sys.path can confuse Python's import system so that lightkurve's
lazy ``from astroquery.exceptions import …`` inside _query_mast fails with
"astroquery is not a package" on Python 3.13.
"""
import astroquery  # noqa: F401
import astroquery.exceptions  # noqa: F401
import astroquery.mast  # noqa: F401
import astroquery.ipac.nexsci.nasa_exoplanet_archive  # noqa: F401
