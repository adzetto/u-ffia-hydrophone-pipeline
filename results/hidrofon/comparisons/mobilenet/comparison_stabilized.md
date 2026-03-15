# Inference Run Comparison

- Baseline run: `raw`

- raw: counts={'weak': 11, 'strong': 3, 'none': 1}, mean_confidence=0.9833
- hum_filtered: counts={'weak': 11, 'strong': 2, 'none': 1, 'medium': 1}, mean_confidence=0.9441
- hum_filtered_adapted: counts={'weak': 9, 'none': 1, 'strong': 5}, mean_confidence=0.9459
- hum_filtered_gated: counts={'weak': 10, 'none': 5}, mean_confidence=1.0000
- hum_filtered_adapted_gated: counts={'weak': 10, 'none': 5}, mean_confidence=0.9889

## Label Changes: hum_filtered vs raw

- kayit_20260313_052719.wav: strong (0.9516) -> medium (0.5831)

## Label Changes: hum_filtered_adapted vs raw

- kayit_20260313_052218.wav: weak (0.9987) -> none (0.9127)
- kayit_20260313_052420.wav: weak (0.9957) -> strong (0.9514)
- kayit_20260313_052609.wav: none (0.9989) -> strong (0.8350)

## Label Changes: hum_filtered_gated vs raw

- kayit_20260313_052420.wav: weak (0.9957) -> none (1.0000)
- kayit_20260313_052517.wav: strong (0.9945) -> none (1.0000)
- kayit_20260313_052719.wav: strong (0.9516) -> none (1.0000)
- kayit_20260313_052822.wav: strong (0.8208) -> none (1.0000)

## Label Changes: hum_filtered_adapted_gated vs raw

- kayit_20260313_052420.wav: weak (0.9957) -> none (1.0000)
- kayit_20260313_052517.wav: strong (0.9945) -> none (1.0000)
- kayit_20260313_052719.wav: strong (0.9516) -> none (1.0000)
- kayit_20260313_052822.wav: strong (0.8208) -> none (1.0000)