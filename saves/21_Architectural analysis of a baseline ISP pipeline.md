## Architectural analysis of a baseline ISP pipeline
- Authors : Park, Hyun Sang
- Journal : Theory and Applications of Smart Cameras
- Year : 2016
- Link : [https://scholar.archive.org...](https://scholar.archive.org/work/3dr4w4hfsnhtnejgcvwgsuenya/access/wayback/https://www.springer.com/cda/content/document/cda_downloaddocument/9789401799867-c2.pdf?SGWID=0-0-45-1518310-p177376119)


### Abstract
- A number of functions are incorporated in an ISP
- ISP functions are divided into pixel-based and frame-based ones, and are dedicated to one of three color domains in Bayer, RGB, or YCbCr.

### Introduction
- Pixel-based Functions    
➔ Utilizing an input pixel and its surrounding pixels
➔ Exploiting spatial information: spatial filter

- Frame-based Functions
➔ Requiring the whole pixels of an image.
➔ Divided by how many images are exploited to get the outcome.

- Global features
➔ Dynamic range extension, Auto-white balance, Auto-exposure, Contrast enhancement

- Temporal correlation
➔ Noise reduction, Rolling-shutter removal, Image stabilization

- Traditional ISPs
➔  Limited frame-based functions (Auto-exposure, Auto-white balance, Auto-focus (3A))

- Proposed baseline ISP pipeline
<img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/415cd05c-5f6f-4023-921a-49a6a64d4556' width=85%>


-------------------

### 1. Embedded ISP Inside an AP (Application Processor)
- The AP provides abundant memory space as well as bandwidth.
- So the pixel-based functions can be processed with a legacy baseline ISP, while the frame-based functions can be processed by programming GPU/GPGPU.
- ISP implementation consumes much energy since it uses the power-hungry memory device and the hot computing units.
➔ It can provide the best quaility of an image for end-user satisfaction.

### 2. Primary ISP Architecture for Bayer Image Sensors
- The ISP contains three components: quantization, color space conversion and data formatter
- The image sensor is assumed to produce analog R, G, and B signals at every pixel position.
  - Y, C<sub>R</sub>, and C<sub>B</sub> signals are calculated trom these digital R. G and B signals.
- While ISP itself isn't standardized, the standardization of digital video, particularly in Rec. ITU-R BT.601 and Rec. ITU-R BT.656 since 1982, includes basic components of an ISP.
- Rec. ITU-R BT.601: This standard focuses on studio encoding parameters for digital television, defining regulations for digitizing SDTV video with a resolution of 720 × 480 or 720 × 576 at a 13.5 MHz sampling frequency.
- **Quantization** Formula: The 8-bit quantization formula, specified in Rec. ITU-R BT.601, converts analog R, G, and B signals to digital RGB signals, ensuring consistency in calculation results across different implementations.

- **Color Space Conversion**: Rec. ITU-R BT.601 provides a specific formula for converting digital R-G-B signals to Y-C<sub>R</sub>-C<sub>B</sub> signals, emphasizing the importance of adhering to the recommended formula for color compatibility among different implementations.
  - ITU-R BT.601 regulates Y-CR-CB subsampling formats like 4:4:4 and 4:2:2, chosen for effective data reduction without significant visual quality loss.

- The ISP's **Data Formatter** handles subsampling and interleaving of Y-CR-CB signals, supporting the standardized 4:2:2 chroma subsampling format.
  - Timing Reference in Rec. ITU-R BT.656: Timing reference signals in video data, derived from reserved codewords (SAV and EAV), ensure proper synchronization between transmitter and receiver.

- **Color Filter Arrays (CFAs) and Bayer Patterns**
➔ Image sensors, often using Bayer patterns for spatial color subsampling, necessitate interpolation (demosaicing) to restore deficient color components.

- **Edge-Directed Interpolation**
➔ Adaptive interpolation techniques, like edge-directed interpolation, help reduce artifacts like pseudo-color and zipper noise in the demosaicing process.

- **Anti-Aliasing Noise Filter in ISP**
➔  An ISP evolved for Bayer sensors addresses artifacts with functions like anti-aliasing noise filter, compensating for challenges posed by Bayer array sensors.

- ISP architecture to recover artifacts from a Bayer image sensor
  - **Anti-aliasing Noise Filter**: Salt-and-pepper noise produced
during the manufacturing of image sensor has to be removed before color interpolation.
  - **Color Filter Array Interpolation**: Restore the original color components from the sampled ones.
➔ It results in zipper noise and pseudo-color. The zipper noise can be suppressed considering edge direction during color interpolation process.
  - **Noise Filter for Luma**: In an anti-aliasing noise filter, it is not possible to exploit correlation with the
adjacent pixels because they are of different color attributes. 
  - **Noise Filter for Chrominance**: Removing pseudo-color caused by subsampling and interpolation
process. 
➔ Because human eyes are very sensitive to rapid color changes, it is necessary to build a natural image by suppressing excessive color changes.

<img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/d93d1545-e089-4db2-83cb-5abe422cd182' width=70%>

### 3.  ISP Architecture for Color Reproduction
- **Color Perception Difference**
➔ Due to the varying responses of silicon sensors and human eyes to light, a process for restoring natural color is essential.
  - Color spaces like CIERGB, CIEXYZ, and sRGB are used to represent colors in a 3D system, with sRGB being widely used in consumer electronics.

- **Gamma Correction**
➔ Nonlinear gamma correction is crucial for adapting the linear response of image sensors to the nonlinear perception of human eyes.

- Defining an **RGB color space** involves specifying red, green, blue primaries, a white point, and a gamma correction curve.
  - An ISP pipeline includes both color correction (linear) and gamma correction (nonlinear), supporting a specific RGB color space.

- **Auto-White Balance (AWB)**
➔ AWB compensates for color distortion due to varying light spectra, adjusting color temperature to match D65 illumination.

- Chromaticity, represented by hue and saturation, is crucial for color perception. ISP performs hue/saturation control in the Y-CR-CB domain.

- Color Domain Functions in ISP
➔ AWB in Bayer, gamma/color correction in RGB, and hue/saturation control in Y-CR-CB domains collectively reproduce accurate colors perceived by human eyes.

- ISP architecture for color reproduction
<img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/45084c16-5bfe-423d-9362-8728542ef212' width=70%>

### 4. ISP Architecture with Pre-/Post-processing
- Pre-Processing Functions
➔ Additional pre-processing compensates for sensor distortions, including dead pixel concealment (DPC) to handle defective pixels.

- **Black Level Compensation (BLC)**
➔ BLC corrects non-linear sensor responses, estimating the sensor response in no-light conditions by subtracting optical black area averages.

- **Lens-Shading Correction (LSC)**
➔ LSC compensates for shading effects caused by lens systems, ensuring consistent light-to-voltage gain for all pixels.

- **Flat Field Compensation (FFC)**
➔ FFC, a type of LSC, compensates for shading by correction gain estimated and stored in advance, improving image uniformity.

- **Noise Reduction**
➔ Noise reduction, crucial for high-resolution sensors, is performed after achieving consistent linearity and is a key factor determining camera system performance.

- **Visual Quality Enhancement**
➔ Techniques like edge enhancement and contrast control are applied post-processing for subjective visual quality improvement, with considerations for potential artifacts.

- ISP architecture for handling sensor derating factors
<img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/abe21307-da8d-4883-ba4d-eda8d19519d1' width=70%>


### 5. Further Works on ISP

- For a legacy ISP pipeline, addressing color-related functions to ensure robust color quality across ambient color temperatures is crucial. 
  - Global information might require significant memory, which can be achieved with external SDRAM for more sophisticated functions.

- Ongoing improvements in color interpolation and noise reduction are key focus areas for ISP development. Additionally, suppressing false colors or removing pseudo-colors has become a significant function to prevent distortion in human perception.
