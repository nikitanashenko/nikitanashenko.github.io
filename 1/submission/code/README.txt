--------------------------------------------------
REQUIREMENTS
--------------------------------------------------
- Python 3.9+
- numpy
- opencv-python

Install with:
    pip install numpy opencv-python

--------------------------------------------------
HOW TO RUN
--------------------------------------------------
1. Place the input images (.jpg and .tif glass plate scans) in the same
   directory as main.py.

2. Make sure the filenames in the `names` list near the bottom of main.py
   match the input files you want to process.

3. Run:
       python main.py

4. Aligned and unaligned color images are written to an `output/` folder
   (created automatically) as:
       out_aligned_<name>.jpg
       out_unaligned_<name>.jpg

*The computed (dx, dy) displacement vectors for the green and red channels
are printed to the terminal for each image.