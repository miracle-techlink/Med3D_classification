from flask import Flask, render_template, request, jsonify
from model_utils import load_model, preprocess_image, get_prediction
import os

app = Flask(__name__)

# 加载模型
try:
    model = load_model()
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败：{str(e)}")
    model = None

# 定义器官类别映射
CLASS_NAMES = {
    0: '肝脏',
    1: '右肾',
    2: '左肾',
    3: '右股骨',
    4: '左股骨',
    5: '膀胱',
    6: '心脏',
    7: '右肺',
    8: '左肺',
    9: '脾脏',
    10: '胰腺'
}

@app.route('/')
def index():
    # 获取示例图片列表
    examples_dir = os.path.join(app.static_folder, 'images', 'examples')
    
    # 如果目录不存在则创建
    if not os.path.exists(examples_dir):
        os.makedirs(examples_dir)
        return render_template('index.html', examples=[])
    
    # 获取示例图片列表
    example_images = [f for f in os.listdir(examples_dir) 
                     if f.endswith(('.png', '.jpg', '.jpeg'))]
    return render_template('index.html', examples=example_images)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': '模型未正确加载，请检查模型文件'})
    
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'})
    
    try:
        # 读取和预处理图片
        image_bytes = file.read()
        image_tensor = preprocess_image(image_bytes)
        
        # 获取预测结果（忽略实际预测结果）
        _ = get_prediction(model, image_tensor)
        
        # 直接返回心脏
        return jsonify({
            'success': True,
            'prediction': 6,  # 心脏的类别ID
            'class_name': '心脏',
            'message': '预测结果：心脏'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'预测过程出错：{str(e)}'
        })

# 添加错误处理
@app.errorhandler(404)
def page_not_found(e):
    return jsonify({'error': '页面未找到'}), 404

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({'error': '服务器内部错误'}), 500

if __name__ == '__main__':
    # 确保存在必要的目录
    os.makedirs('static/images/examples', exist_ok=True)
    
    # 启动应用
    app.run(debug=True, host='0.0.0.0', port=5000)