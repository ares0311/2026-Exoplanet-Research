# Exoplanet Detection and Research: Satellite, AI, and Literature Brief for Coding Agents

Last updated: 2026-06-26

## Mission Objective

Build software that detects, triages, validates, and characterizes exoplanets using public light curves, target-pixel files, spectra, catalogs, and atmospheric-retrieval workflows. The dominant software stack should support transit photometry, false-positive vetting, ephemeris refinement, atmospheric retrieval, and future direct-imaging target preparation.

## Ranked Space Assets and Satellites

Ranking criteria: exoplanet-specific impact, current/future data value, open archive quality, detection/characterization power, and usefulness for coding projects.

| Rank | Asset | Status | Best Use | Why It Ranks Here | Data / Access |
|---:|---|---|---|---|---|
| 1 | TESS | Active | All-sky transit detection around bright nearby stars | Best current public discovery engine for coding work: huge archive, full-frame images, target-pixel files, light curves, TOIs, and ongoing sectors. | MAST, NASA Exoplanet Archive, ExoFOP-TESS. |
| 2 | Kepler / K2 | Completed | Benchmark transit detection, occurrence rates, ML training | Highest-value historical benchmark for small transiting planets and ML methods. Kepler produced cleaner long-baseline light curves; K2 adds noisier, more realistic systematics. | MAST, NASA Exoplanet Archive. |
| 3 | JWST | Active | Atmospheric spectroscopy and high-precision follow-up | Most powerful current exoplanet atmosphere characterization observatory. Not a bulk detector, but central to modern research. | MAST public archive after proprietary period; GO proposals for new data. |
| 4 | ESA PLATO | Future, launch planned end-2026 per mission paper | Bright-star terrestrial planets, asteroseismology, ages | Likely the next major transit mission for well-characterized small planets around bright stars, with stellar ages from asteroseismology. | ESA archive after launch; proposal/community products TBD. |
| 5 | Nancy Grace Roman Space Telescope | Future | Microlensing planet census, direct-imaging tech demo, IR surveys | Roman will provide a statistical census through microlensing and a coronagraph technology pathfinder for direct imaging. | NASA archive after launch; proposal and survey products. |
| 6 | CHEOPS | Active | Follow-up photometry and radius refinement | High-precision follow-up of known planets, especially bright hosts. Valuable for ephemeris/radius refinement rather than bulk discovery. | ESA archive; guest observer programs. |
| 7 | Ariel | Future | Atmosphere population survey | Designed to characterize exoplanet atmospheres statistically, especially warm/hot planets. Critical for ML atmospheric retrieval benchmarks. | ESA archive after launch; Ariel ML challenge datasets already useful. |
| 8 | Hubble | Active/legacy | Transit/eclipse spectroscopy, legacy atmospheres | Still important for comparative atmospheric work and UV/optical follow-up, though lower throughput than JWST. | MAST. |
| 9 | Gaia | Archive continuing | Astrometry, host-star parameters, giant exoplanets, target vetting | Gaia parallaxes and stellar parameters underpin exoplanet radius/mass inference; astrometric planets are a growing contribution. | ESA Gaia Archive. |
| 10 | Spitzer | Completed | Infrared transit/eclipse legacy data | Key historical IR atmosphere and phase-curve archive; useful for retrieval benchmarking. | IRSA. |
| 11 | CoRoT | Completed | Early space transit survey | Historically important; less central than Kepler/TESS but relevant for archive completeness and method history. | CNES/ESA-associated archives. |
| 12 | Euclid | Active | Microlensing synergy, host characterization, possible transits in fields | Not an exoplanet mission, but useful for Roman microlensing synergy and stellar/galaxy context. | ESA archive. |
| 13 | Habitable Worlds Observatory | Concept/development | Future direct imaging of Earth-like exoplanets | Not available now, but culturally important: it defines the long-term direct-imaging/biosignature target. | No operational archive yet. |

## Credentials and Access Requirements

| Task | Credentials Needed | Notes |
|---|---|---|
| Download TESS, Kepler, K2 light curves and target-pixel files | Usually none | Use MAST web, `astroquery.mast`, `lightkurve`, and NASA Exoplanet Archive tables. |
| Query NASA Exoplanet Archive | Usually none | Good for confirmed planets, candidates, stellar properties, TOIs, KOIs, TAP queries. |
| Use ExoFOP-TESS | Web access; account useful for contributions | Public candidate/follow-up data are visible; contributing follow-up requires account/community participation. |
| Download JWST/Hubble public data | None for public data; MAST account useful | New observations require peer-reviewed proposals. |
| Use ESA CHEOPS/PLATO/Ariel data | Public data after release; proposal access for new observations | CHEOPS observing time is proposal-based; PLATO/Ariel details depend on mission operations. |
| Contribute confirmed exoplanets | Scientific publication/follow-up evidence | Confirmation is cultural, not just technical: needs vetting, false-positive rejection, often RV or high-resolution imaging. |
| Use Gaia stellar context | None for most queries | Registered Gaia account helps with long async jobs and user tables. |

## Frontier AI and Computational Methods

| Method | Implementation Target | Why It Is Used | Key Sources |
|---|---|---|---|
| CNN classifiers for phase-folded transits | Kepler, K2, TESS TCE/TOI triage | CNNs learn transit shape, secondary eclipses, odd/even depth, and local/global context faster than manual vetting. | Shallue & Vanderburg 2017: https://arxiv.org/abs/1712.05044; Osborn et al. 2019: https://arxiv.org/abs/1902.08544 |
| Astronet-style TESS triage | TESS FFI candidate ranking | Automates triage of huge TESS candidate streams and helps avoid losing real candidates in noisy data. | Tey et al. 2023: https://arxiv.org/abs/2301.01371 |
| Transformer / attention models for full light curves | TESS full-frame image light curves | Attention can model long light curves without hand-selecting transit windows and may improve instrument-agnostic triage. | TESS full-light-curve transformer example: https://arxiv.org/html/2502.07542v1 |
| Semi-supervised / unsupervised anomaly detection | TESS and Kepler candidate mining | Useful when labels are incomplete and rare events are underrepresented. Helps discover unusual systems and false-positive classes. | Ofman et al. 2021: https://arxiv.org/abs/2102.10326 |
| Gaussian Processes for stellar variability/systematics | Transit fitting, radial velocities, rotation, stellar activity | GPs model correlated noise and stellar variability, reducing biased transit depth/timing estimates. | Aigrain & Foreman-Mackey review: https://arxiv.org/abs/2209.08940; Angus et al. 2017: https://arxiv.org/abs/1706.05459; Barros et al. 2020: https://arxiv.org/abs/2001.07975 |
| Bayesian atmospheric retrieval | JWST/Hubble/Spitzer/Ariel spectra | Standard method for inferring molecular abundances, temperature structure, clouds, and uncertainties from spectra. | Retrieval culture baseline: TauREx, petitRADTRANS, CHIMERA literature; ML retrieval comparisons below. |
| Random Forest / Bayesian neural network atmospheric retrieval | Fast approximate retrieval | Replaces or accelerates expensive nested sampling when many spectra must be processed. | Márquez-Neila et al. 2018: https://arxiv.org/abs/1806.03944; Cobb et al. 2019: https://arxiv.org/abs/1905.10659; Nixon & Madhusudhan 2020: https://arxiv.org/abs/2004.10755 |
| Simulation-based inference / neural posterior estimation | Self-consistent atmospheric models | Learns posterior distributions from simulations, enabling faster inference with more realistic forward models. | Martínez et al. 2024: https://arxiv.org/abs/2401.04168 |
| ML cross-correlation spectroscopy | High-resolution spectra | Improves detection of weak molecular signatures by learning patterns in cross-correlation space. | Garvin et al. 2024: https://arxiv.org/abs/2405.13469 |
| Direct-imaging ML / transformer methods | Coronagraphic planet detection | Helps separate faint planets from speckles and residual starlight in high-contrast imaging. | Transformer direct-imaging direction: https://arxiv.org/abs/2508.14508 |

## Influential and Innovative arXiv Papers

| Paper | Why It Matters | Coding-Agent Takeaway |
|---|---|---|
| Ricker et al., “The Transiting Exoplanet Survey Satellite” https://arxiv.org/abs/1406.0151 | Mission-defining TESS paper, cited heavily. | Use it to understand TESS observing strategy, cadence, and target logic. |
| Sullivan et al., TESS yield simulations https://arxiv.org/abs/1506.03845 | Influential prediction of TESS planet yield. | Use for synthetic populations and expected detection distributions. |
| Shallue & Vanderburg, “Identifying Exoplanets with Deep Learning” https://arxiv.org/abs/1712.05044 | Landmark deep-learning exoplanet paper; recovered planets in Kepler systems. | Baseline architecture for transit vetting and explainability via local/global views. |
| Tey et al., “Identifying Exoplanets with Deep Learning V” https://arxiv.org/abs/2301.01371 | Modern TESS FFI triage model used in operational-style workflows. | Good model/data organization template for TESS candidate classifiers. |
| Rauer et al., “The PLATO Mission” https://arxiv.org/abs/2406.05447 | Defines PLATO’s science: bright stars, terrestrial planets, precise ages. | Prepare code for PLATO-style long-baseline photometry and asteroseismic context. |
| Benz et al., “The CHEOPS mission” https://arxiv.org/abs/2009.11633 | Defines CHEOPS follow-up photometry role and precision. | Use CHEOPS as a radius/ephemeris refinement asset, not a bulk discovery survey. |
| Akeson et al., “The NASA Exoplanet Archive” https://arxiv.org/abs/1307.2944 | Describes a core database used across the field. | Build archive adapters around NASA Exoplanet Archive TAP/table conventions. |
| Aigrain & Foreman-Mackey, “Gaussian Process regression for astronomical time-series” https://arxiv.org/abs/2209.08940 | Practical GP review with exoplanet relevance. | Use for correlated-noise modeling and avoid overconfident transit/RV fits. |
| Márquez-Neila et al., “Supervised Machine Learning for Analysing Spectra of Exoplanetary Atmospheres” https://arxiv.org/abs/1806.03944 | Early influential ML atmospheric retrieval work. | Use as baseline for fast retrieval from simulated grids. |
| Martínez et al., “Enabling self-consistent exoplanet atmospheric retrievals with simulation-based inference” https://arxiv.org/abs/2401.04168 | Frontier direction for neural posterior estimation. | Treat SBI as a serious path for JWST/Ariel-scale retrieval pipelines. |

## Citizen Candidate Vetting and Submission Best Practices

Goal: contribute useful evidence to the exoplanet community without overstating a detection. Citizen observers are most valuable for follow-up photometry, ephemeris maintenance, false-positive checks, and structured vetting of TESS candidates. New exoplanet confirmation is a community process, usually requiring multiple data types.

### Best Submission Path

| Scenario | Best Path | Practical Notes |
|---|---|---|
| You observed a known transiting exoplanet | Reduce with EXOTIC, AstroImageJ, HOPS, or equivalent; submit to AAVSO Exoplanet Database / Exoplanet Watch workflow | Useful for ephemeris maintenance and transit timing. |
| You followed a TESS Object of Interest (TOI) | Upload TFOP-related follow-up data to ExoFOP-TESS if you are part of/working with TFOP; otherwise coordinate through AAVSO/Exoplanet Watch/experienced TFOP members | ExoFOP is the main community coordination repository for TOI follow-up. |
| You found a new transit-like signal in TESS/Kepler data | First run false-positive vetting and check TOI/KOI/CTOI catalogs; share reproducible notebooks and candidate packets with relevant community channels | A transit-like dip is not enough; eclipsing binaries and systematics dominate. |
| You want structured citizen-science vetting | Use Planet Hunters / Planet Patrol style projects when active | These projects provide common labels and expert review. |
| You have small-telescope data | Focus on high-SNR known targets, bright TOIs, timing refinement, and nearby contaminant checks | Citizen small telescopes are scientifically useful when cadence, timing, and precision are controlled. |

### Minimum Evidence Package

| Item | Required Practice |
|---|---|
| Target identity | TIC/KIC/EPIC/Gaia IDs, coordinates, magnitude, known aliases. |
| Observation metadata | Site, telescope, camera, filter, exposure, cadence, airmass range, seeing if available, UTC times. |
| Calibration | Bias/dark/flat workflow, aperture settings, comparison stars, detrending parameters. |
| Light curve | Time, normalized flux, flux error, filter, flags, and unbinned plus binned versions. |
| Transit model | Period/epoch assumptions, fitted mid-transit time, depth, duration, impact parameter if modeled, uncertainties. |
| Baseline | Pre-ingress and post-egress coverage; partial transits are lower value unless target is high priority. |
| False-positive checks | Nearby eclipsing binaries, centroid shift, odd/even depth, secondary eclipse, Gaia companions/RUWE, contamination ratio. |
| Reproducibility | Raw/calibrated images or archive links, reduction script/notebook, software versions. |

### Citizen Workflow for Follow-up Photometry

1. Pick targets from Exoplanet Watch target lists, ExoFOP-TESS priorities, AAVSO campaigns, or TFOP coordination.
2. Use NASA Exoplanet Archive ephemeris tools or Swarthmore Transit Finder to confirm observability from your site.
3. Observe enough baseline before and after transit; full transits are far more useful than partial dips.
4. Keep exposures in the linear detector regime and avoid saturation of target or comparison stars.
5. Use stable comparison stars with similar color/brightness when possible.
6. Reduce consistently in EXOTIC, AstroImageJ, HOPS, or a documented Python pipeline.
7. Inspect residuals for clouds, meridian flips, guiding loss, focus drift, and airmass trends.
8. Submit to AAVSO Exoplanet Database / Exoplanet Watch when using that workflow; upload/share through ExoFOP-TESS when participating in TFOP-related follow-up.
9. For a new candidate, publish a candidate packet, not a claim: light curve, vetting diagnostics, catalog checks, and suggested follow-up.

### Quality Bar Before Asking for Community Follow-up

| Green Flag | Red Flag |
|---|---|
| Repeatable periodic transit-like signal | One isolated dip with no recurrence |
| Full transit with good baseline | Partial event with no out-of-transit baseline |
| Signal survives different detrending choices | Signal appears only after aggressive detrending |
| No nearby Gaia/eclipsing binary contaminant | Bright neighbor inside aperture or centroid shift |
| Plausible planet radius/duration for host star | Implied companion is stellar-sized or duration impossible |
| Clear uncertainty estimates | Pretty plot without machine-readable data |

### What Not To Do

- Do not call a TESS threshold-crossing event a planet before vetting.
- Do not submit only screenshots; submit machine-readable light curves.
- Do not mix time standards; use BJD_TDB when the reduction tool supports it, and document it.
- Do not ignore nearby stars inside the aperture.
- Do not over-detrend away astrophysical signals or create false dips.

### Source Pointers

- TESS Follow-up Observing Program: https://heasarc.gsfc.nasa.gov/docs/tess/tfop.html
- TESS follow-up / ExoFOP-TESS submission: https://tess.mit.edu/followup/exofop-tess/
- ExoFOP project description: https://www.ipac.caltech.edu/project/exofop
- Join TFOP: https://tess.mit.edu/followup/apply-join-tfop/
- AAVSO Exoplanet Section: https://www.aavso.org/exoplanet-section
- AAVSO exoplanet report format: https://www.aavso.org/aavso-exoplanet-report-file-format
- Exoplanet Watch submit data: https://science.nasa.gov/citizen-science/exoplanet-watch/how-to-contribute/how-to-submit-your-data/
- Zellem et al., small telescopes for exoplanet follow-up: https://arxiv.org/abs/2003.09046
- NASA Exoplanet Archive and ExoFOP overview: https://arxiv.org/abs/2506.03299

## Coding-Agent Guidance

1. Build ingestion for MAST/TESS/Kepler first, then NASA Exoplanet Archive and Gaia host-star enrichment.
2. Keep detection, vetting, validation, and characterization as separate pipeline stages.
3. Every candidate needs false-positive features: secondary eclipse, odd/even depths, centroid shifts, nearby contamination, stellar variability, Gaia RUWE/companions, and ephemeris consistency.
4. Use `lightkurve`, `astroquery`, `eleanor`, `wotan`, `transitleastsquares`, `exoplanet`, `celerite`, and `pymc` where appropriate.
5. For atmospheric work, separate forward model, likelihood, sampler/inference engine, and posterior diagnostics.

## Source URLs

- TESS MIT: https://tess.mit.edu/
- TESS MAST: https://archive.stsci.edu/missions-and-data/tess
- Kepler MAST: https://archive.stsci.edu/missions-and-data/kepler
- NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu/
- NASA ExEP missions: https://science.nasa.gov/astrophysics/programs/exep/missions/
- JWST exoplanets: https://science.nasa.gov/mission/webb/science-overview/science-explainers/webbs-impact-on-exoplanet-research/
- ESA CHEOPS/PLATO/Ariel context: https://www.esa.int/ESA_Multimedia/Videos/2023/03/ESA_s_exoplanet_missions
- Roman: https://science.nasa.gov/mission/roman-space-telescope/
- HWO: https://science.nasa.gov/astrophysics/programs/habitable-worlds-observatory/
