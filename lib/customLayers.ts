import * as tf from '@tensorflow/tfjs';

/** L2-normalize embeddings along the last axis (matches training Lambda). */
export class L2NormalizeLayer extends tf.layers.Layer {
  static className = 'L2NormalizeLayer';

  constructor(config?: any) {
    super(config ?? {});
  }

  computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
    return inputShape;
  }

  call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
    return tf.tidy(() => {
      const input = Array.isArray(inputs) ? inputs[0] : inputs;
      const norm = input.norm(2, -1, true);
      return input.div(norm.add(tf.scalar(1e-12)));
    });
  }

  getConfig(): any {
    return super.getConfig();
  }
}

/** Element-wise L1 distance between two embedding vectors. */
export class L1DistanceLayer extends tf.layers.Layer {
  static className = 'L1DistanceLayer';

  constructor(config?: any) {
    super(config ?? {});
  }

  computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
    if (Array.isArray(inputShape) && Array.isArray(inputShape[0])) {
      return inputShape[0];
    }
    return inputShape;
  }

  call(inputs: tf.Tensor | tf.Tensor[]): tf.Tensor {
    return tf.tidy(() => {
      const [a, b] = inputs as tf.Tensor[];
      return tf.abs(tf.sub(a, b));
    });
  }

  getConfig(): any {
    return super.getConfig();
  }
}

let registered = false;

export function registerCustomLayers(): void {
  if (registered) return;
  tf.serialization.registerClass(L2NormalizeLayer);
  tf.serialization.registerClass(L1DistanceLayer);
  registered = true;
}
