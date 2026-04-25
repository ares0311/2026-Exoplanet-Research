# SCORING MODEL

## Bayesian Framework

P(H|D) = P(D|H) * P(H) / normalization

## Hypotheses
- Planet candidate
- Eclipsing binary
- Background eclipsing binary
- Stellar variability
- Instrumental artifact

## Features
- SNR
- Transit count
- Depth consistency
- Duration plausibility
- Odd/even mismatch
- Secondary eclipse
- Contamination risk

## Logistic Approximation

logit(p_planet) =
    + 1.2 * log(SNR)
    + 0.9 * transit_count_score
    + 0.7 * duration_score
    - 1.5 * odd_even
    - 1.8 * secondary_eclipse
    - 1.2 * contamination

p_planet = sigmoid(logit)

## Outputs
- p_planet_candidate
- false_positive_probability
- detection_confidence
