****Wavelet Edge Detector GUI****

Visualizes an approach to detect the edges of an image using multiple levels of wavelet decomposition and then reconstruction.
- The wavelet string input field can be a name in the set of all DWT wavelets supported by pywt.
- The '+' and '-' buttons can add and subtract respectively, a level of decomposition of the highest LL subband decomposition until then. The type of wavelet decomposition is the wavelet string input at that moment.
- Adding or subtracting levels of decomposition corresponds to adding or subtracting sliders in the range 0.00 to 1.00. These control the weights of the detail subbands (HL, LH, HH) during reconstruction (Inverse DWT). Additionally, the detail subbands are also multiplied by the pixel-wise coherence masks based on interpreting the HL, LH subbands as gradients along x, y directions. This is done for each level. The highest LL subband is zeroed out during sequential (highest to lowest level) reconstruction.
- Original grayscale (image automatically converted to grayscale) image is displayed at the left and the edges are at the right. Controls are the bottom. Beside each slider, the level number and the wavelet used at that decomposition level is depicted in the format: L{level_number}[{wavelet}] weight

To run the GUI, follow the format:
python wavelet_edge_extractor.py "image_filename.jpg"
