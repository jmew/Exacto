# ImageBackgroundRemover
A tool to do foreground extraction in batches leaving behind a white background where the original background used to be. It is meant to be used in to help with generating datasets for training neural networks through removing unwanted things in the background of the image.

## Usage
python background_remover.py [IMG_SOURCE_DIR] [IMG_OUTPUT_DIR]

e.g
```bash
python background_remover.py images/ output/
```

## Examples

**Before**

![shoes with background](https://i.imgur.com/Y3NVyP7.jpg)

**After**

![shoes without background](https://i.imgur.com/XEWvbhc.jpg)

---

**Before**

![shoes with background](https://i.imgur.com/mxkmfI4.jpg)

**After**

![shoes without background](https://i.imgur.com/feYynaR.jpg)
