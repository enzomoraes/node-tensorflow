import * as fs from 'fs';
import path from 'path';

export class MNIST {
  train_images: Image[] = [];
  train_labels: Label[] = [];
  test_images: Image[] = [];
  test_labels: Label[] = [];
}

export class Image {
  constructor(
    public rows: number,
    public columns: number,
    public pixels: number[]
  ) {}

  display(): void {
    let result = `Image (${this.rows} x ${this.columns})\n`;

    for (let i = 0; i < this.rows * this.columns; i++) {
      if (i % this.columns === 0) {
        result += '\n';
      }
      result +=
        this.pixels[i] * 255.0 === 0
          ? ' '
          : this.pixels[i] * 255.0 === 255
          ? '█'
          : '#';
    }

    console.log(result);
  }
}

export class Label {
  constructor(public label: number) {}

  display(): void {
    console.log(`Label: ${this.label}`);
  }
}

async function readUInt32FromFile(
  file: fs.promises.FileHandle
): Promise<number> {
  const buffer = Buffer.alloc(4);
  await file.read(buffer, 0, 4, null);
  return buffer.readUInt32BE(0);
}

async function readUInt8FromFile(
  file: fs.promises.FileHandle
): Promise<number> {
  const buffer = Buffer.alloc(1);
  await file.read(buffer, 0, 1, null);
  return buffer.readUInt8(0);
}

async function readImages(filename: string): Promise<Image[]> {
  const file = await fs.promises.open(filename, 'r');
  const magicNumber = await readUInt32FromFile(file);
  const numImages = await readUInt32FromFile(file);
  const numRows = await readUInt32FromFile(file);
  const numColumns = await readUInt32FromFile(file);

  const images: Image[] = [];

  for (let i = 0; i < numImages; i++) {
    const pixelsBuffer = Buffer.alloc(numRows * numColumns);
    await file.read(pixelsBuffer, 0, numRows * numColumns, null);
    const pixels = Array.from(pixelsBuffer).map(p => p / 255.0);
    images.push(new Image(numRows, numColumns, pixels));
  }

  await file.close();

  return images;
}

async function readLabels(filename: string): Promise<Label[]> {
  const file = await fs.promises.open(filename, 'r');
  const magicNumber = await readUInt32FromFile(file);
  const numLabels = await readUInt32FromFile(file);

  const labels: Label[] = [];

  for (let i = 0; i < numLabels; i++) {
    const label = await readUInt8FromFile(file);
    labels.push(new Label(label));
  }

  await file.close();

  return labels;
}

export async function loadMNIST(): Promise<MNIST> {
  const mnist = new MNIST();
  const dataDir = path.join(__dirname, '..', 'data'); // Adjust the path accordingly

  mnist.train_images = await readImages(`${dataDir}/train-images.idx3-ubyte`);
  mnist.train_labels = await readLabels(`${dataDir}/train-labels.idx1-ubyte`);
  mnist.test_images = await readImages(`${dataDir}/t10k-images.idx3-ubyte`);
  mnist.test_labels = await readLabels(`${dataDir}/t10k-labels.idx1-ubyte`);

  return mnist;
}
