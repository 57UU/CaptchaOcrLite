import onnxruntime as ort
import numpy as np
from PIL import Image

num2char = {
    k:v for k,v in enumerate("0123456789abcdefghijklmnopqrstuvwxyz")
}
char_length = len(num2char)

class CaptchaONNXInference:
    def __init__(self, model_path="export/captcha_model.onnx", providers=None):
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize((80, 26))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.transpose(img_array, (1, 2, 0))
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def preprocess_array(self, img_array):
        if img_array.max() > 1:
            img_array = img_array.astype(np.float32) / 255.0
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def predict(self, input_data):
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        output = outputs[0]
        result = []
        for i in range(4):
            char_logits = output[0, i * char_length:(i + 1) * char_length]
            char_index = np.argmax(char_logits)
            result.append(num2char[char_index])
        return "".join(result)
    
    def predict_from_file(self, image_path):
        input_data = self.preprocess_image(image_path)
        return self.predict(input_data)
    
    def predict_batch(self, input_data):
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        output = outputs[0]
        batch_size = output.shape[0]
        results = []
        for b in range(batch_size):
            result = []
            for i in range(4):
                char_logits = output[b, i * char_length:(i + 1) * char_length]
                char_index = np.argmax(char_logits)
                result.append(num2char[char_index])
            results.append("".join(result))
        return results

if __name__ == "__main__":
    import sys
    
    model = CaptchaONNXInference()
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = model.predict_from_file(image_path)
        print(f"预测结果: {result}")
    else:
        print("使用方法: python onnx_inference.py <image_path>")
        print("示例: python onnx_inference.py captcha.png")
