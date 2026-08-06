import * as tf from '@tensorflow/tfjs';

export const MODEL_INPUT_HEIGHT = 112;
export const MODEL_INPUT_WIDTH = 896;

export function preprocessImage(imageData: ImageData): tf.Tensor4D {
  return tf.tidy(() => {
    const tensor = tf.browser.fromPixels(imageData);

    const grayscale = tensor.mean(2).toFloat();

    const expanded = grayscale.expandDims(0).expandDims(-1);

    const resized = tf.image.resizeBilinear(expanded as tf.Tensor4D, [
      MODEL_INPUT_HEIGHT,
      MODEL_INPUT_WIDTH,
    ]);

    const normalized = resized.div(255.0);

    return normalized as tf.Tensor4D;
  });
}

export function tileGrayscaleTo3Channel(tensor: tf.Tensor4D): tf.Tensor4D {
  return tf.tidy(() => tf.concat([tensor, tensor, tensor], 3) as tf.Tensor4D);
}

export async function fileToImageData(file: File): Promise<ImageData> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = MODEL_INPUT_WIDTH;
        canvas.height = MODEL_INPUT_HEIGHT;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          reject(new Error('Could not get 2D canvas context'));
          return;
        }
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(0, 0, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);
        const ratio = Math.min(MODEL_INPUT_WIDTH / img.width, MODEL_INPUT_HEIGHT / img.height);
        const w = img.width * ratio;
        const h = img.height * ratio;
        ctx.drawImage(
          img,
          (MODEL_INPUT_WIDTH - w) / 2,
          (MODEL_INPUT_HEIGHT - h) / 2,
          w,
          h,
        );
        try {
          const data = ctx.getImageData(0, 0, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);
          resolve(data);
        } catch (err) {
          reject(err);
        }
      };
      img.onerror = () => reject(new Error('Failed to decode image'));
      img.src = e.target?.result as string;
    };
    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.readAsDataURL(file);
  });
}

export function preprocessForXception(imageData: ImageData): tf.Tensor4D {
  return tf.tidy(() => {
    const gray = preprocessImage(imageData);

    const rgb = gray.concat([gray, gray], 3) as tf.Tensor4D;

    const preprocessed = tf.tidy(() => {
      const scaled = rgb.mul(2.0).sub(1.0);
      return scaled as tf.Tensor4D;
    });

    gray.dispose();
    rgb.dispose();

    return preprocessed;
  });
}
