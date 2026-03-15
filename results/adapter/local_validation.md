# Local Validation Note

## Binary-Gated Local Folder Check

- `PROJE1/voice` -> `333/333` files predicted as `none`
- `PROJE1/voice_yem` -> `235` `strong`, `97` `none`, `1` `weak`

This confirms that the binary gate cleanly suppresses the nonfeeding local folder, but the released checkpoint still does not transfer cleanly inside the positive side of the local domain.

## Exploratory 4-Way Local Projection

- Local folders used: `voice`, `voice2`, `voice_yem`, `voice1`
- Feature family: MFCC + spectral centroid + spectral bandwidth + rolloff + zero-crossing rate + RMS
- Cross-validation accuracy: `0.9970`
- Cross-validation standard deviation: `0.0033`

Hydrophone projection result:

- First `10` hydrophone files -> `voice2`
- Last `5` hydrophone files -> `voice`

Practical interpretation:

- `voice` behaves as the local `none-like` cluster.
- `voice2` behaves as the closest local `weak-like` cluster for the current hydrophone recordings.
- This supports the recommended stabilized hydrophone result of `10 weak / 5 none`.
