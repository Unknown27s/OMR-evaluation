# OMR Evaluation System

An advanced Optical Mark Recognition (OMR) system for automated grading of multiple-choice answer sheets.

## Features

- **Multiple Input Formats**: Supports images (JPG, PNG, TIFF) and PDF files
- **Flexible Answer Keys**: Load answer keys from Excel (.xlsx/.xls) or JSON files
- **Smart Detection Modes**:
  - **Adaptive Contour Mode**: Automatically detects bubbles using contour detection
  - **ROI/Grid Mode**: Template-based scanning with predefined regions
  - **Large Box Mode**: Finds the largest rectangular box and applies grid scanning
- **Batch Processing**: Process multiple sheets simultaneously with parallel workers
- **Detailed Results**: 
  - CSV export with question-by-question analysis
  - Annotated images showing detected answers
  - Confidence scores for each detection
  - Aggregate summary across all sheets
- **Error Detection**: Identifies multiple marks, no marks, and ambiguous answers
- **PDF Support**: Automatically converts PDF pages to images for processing

## Installation

### Prerequisites
- Python 3.8 or higher
- Poppler (for PDF processing)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Installing Poppler (for PDF support)

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils
```

**macOS:**
```bash
brew install poppler
```

**Windows:**
Download from: https://github.com/oschwartz10612/poppler-windows/releases/

## Usage

### Starting the Application

```bash
streamlit run omr_app_multi.py
```

The application will open in your default web browser at `http://localhost:8501`

### Answer Key Format

#### Excel Format (.xlsx/.xls)

Create an Excel file with:
- A column labeled "Answer" containing the correct answers
- Answers can be in format: A, B, C, D or 0, 1, 2, 3
- Optional: Include ROI coordinates (roi_y1, roi_y2, roi_x1, roi_x2) for grid mode

Example:
```
Question | Answer
1        | A
2        | B
3        | C
4        | D
```

#### JSON Format

```json
{
  "num_questions": 50,
  "num_choices": 4,
  "answers": [0, 1, 2, 3, 0, 1, ...],
  "roi": [100, 800, 50, 600]
}
```

For multi-sheet/multi-subject keys:
```json
{
  "Set A": {
    "Math": {
      "num_questions": 30,
      "num_choices": 4,
      "answers": [0, 1, 2, ...]
    },
    "Science": {
      "num_questions": 30,
      "num_choices": 4,
      "answers": [0, 1, 2, ...]
    }
  }
}
```

### Processing Steps

1. **Upload Answer Key**: Select your Excel or JSON answer key file
2. **Upload OMR Sheets**: Upload one or more images or PDF files
3. **Configure Options** (Sidebar):
   - **Scanning Mode**: Choose detection method
   - **Absolute Min Pixels**: Minimum pixels to consider a bubble marked (default: 30)
   - **Ambiguity Ratio**: Threshold for detecting ambiguous marks (default: 0.8)
   - **Debug Images**: Show intermediate processing steps
   - **Parallel Workers**: Number of concurrent processing threads
4. **Start Grading**: Click the button to begin processing
5. **Download Results**: Get ZIP file with annotated images and CSV results

### Output Files

- **{filename}_annotated.png**: Image with detected answers highlighted
  - Green boxes: Correct answers
  - Red boxes: Incorrect answers
  - Cyan boxes: Multiple marks detected
- **{filename}_results.csv**: Detailed results per question
- **aggregate_results.csv**: Summary of all processed sheets

## Scanning Modes

### Adaptive Contour Mode (Recommended)
- Automatically detects bubble positions
- Best for standard OMR sheets with consistent bubble sizes
- No template required
- Handles rotation and slight perspective distortion

### ROI/Grid Mode
- Uses predefined Region of Interest (ROI)
- Best for fixed-template sheets
- Requires ROI coordinates in answer key
- More consistent for identical sheet layouts

### Large Box Mode
- Finds the largest rectangular box on the sheet
- Applies grid scanning within that box
- Good fallback when contour detection fails

## Troubleshooting

### No Bubbles Detected
- Ensure the image is clear and well-lit
- Try increasing image DPI for PDF conversions
- Adjust the "Absolute Min Pixels" threshold
- Use ROI/Grid mode with defined coordinates

### Multiple Marks Detected
- Check for stray marks or incomplete erasures
- Adjust the "Ambiguity Ratio" setting
- Review the annotated image to identify issues

### PDF Processing Fails
- Ensure Poppler is installed and in PATH
- Specify Poppler path in the application if needed
- Try converting PDF to images manually first

## Advanced Configuration

Edit `omr_app_multi.py` to customize:
- Bubble size detection parameters (lines 217-218)
- Contour filtering thresholds
- Grid cell calculations
- Image preprocessing steps

## Dependencies

- streamlit: Web interface
- opencv-python: Image processing
- numpy: Numerical operations
- pandas: Data manipulation and CSV export
- imutils: Image utilities
- pdf2image: PDF conversion
- openpyxl: Excel file reading

## License

MIT License

## Contributing

Contributions are welcome! Please submit pull requests or open issues for bugs and feature requests.
