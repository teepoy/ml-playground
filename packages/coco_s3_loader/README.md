# COCO S3 Loader

A PyTorch-compatible dataloader for loading COCO image datasets from JSON annotation files with S3 URLs as image paths.

## Features

- **S3 URL Support**: Load images directly from S3 URLs specified in COCO JSON annotations
- **Error Handling**: Gracefully handle inaccessible S3 URLs with configurable behavior
- **Transformations**: Apply PyTorch transforms to images during loading
- **Batching**: Full support for PyTorch DataLoader batching
- **Type Safety**: Comprehensive type hints for better IDE support
- **Well Documented**: Detailed docstrings and examples

## Installation

```bash
cd packages/coco_s3_loader
uv sync
```

## Usage

### Basic Example

```python
from coco_s3_loader import CocoS3Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset
dataset = CocoS3Dataset(
    annotation_file='path/to/coco_annotations.json',
    transform=transform,
    handle_errors='skip'  # Options: 'skip', 'raise', 'return_none'
)

# Create dataloader with batching
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Iterate through batches
for batch in dataloader:
    images, annotations = batch
    # Your training code here
    pass
```

### COCO JSON Format

The loader expects COCO-format JSON with S3 URLs in the image paths:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "s3://bucket-name/path/to/image1.jpg",
      "height": 480,
      "width": 640
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 100, 200, 200],
      "area": 40000,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "person",
      "supercategory": "human"
    }
  ]
}
```

### Error Handling Options

- **`skip`**: Skip images that fail to load (default)
- **`raise`**: Raise an exception on load failure
- **`return_none`**: Return None for failed images (useful for debugging)

### AWS Credentials

The loader uses boto3, which automatically uses AWS credentials from:
- Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
- AWS credentials file (`~/.aws/credentials`)
- IAM roles (when running on EC2/ECS)

## API Reference

### CocoS3Dataset

```python
CocoS3Dataset(
    annotation_file: str,
    transform: Optional[Callable] = None,
    handle_errors: str = 'skip',
    s3_client: Optional[boto3.client] = None
)
```

**Parameters:**
- `annotation_file`: Path to COCO JSON annotation file
- `transform`: Optional torchvision transforms to apply to images
- `handle_errors`: How to handle image load errors ('skip', 'raise', 'return_none')
- `s3_client`: Optional pre-configured boto3 S3 client

**Returns:**
- Tuple of (image, annotation_dict) or None if error and `handle_errors='return_none'`

## Testing

Run the test suite:

```bash
cd packages/coco_s3_loader
uv run pytest
```

## License

This package is part of the ml-playground repository.
