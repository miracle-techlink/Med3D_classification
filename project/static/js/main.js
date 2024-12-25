document.addEventListener('DOMContentLoaded', function() {
    // 获取DOM元素
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const preview = document.getElementById('preview');
    const previewImage = document.getElementById('previewImage');
    const result = document.getElementById('result');
    const loading = document.getElementById('loading');

    // 处理拖放事件
    dropZone.addEventListener('dragover', function(e) {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', function(e) {
        e.preventDefault();
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', function(e) {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // 处理点击上传
    dropZone.addEventListener('click', function() {
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            handleFile(this.files[0]);
        }
    });

    // 处理示例图片的拖放
    document.querySelectorAll('.example-item img').forEach(img => {
        img.addEventListener('dragstart', function(e) {
            e.dataTransfer.setData('text/plain', this.src);
        });
    });

    // 文件处理函数
    function handleFile(file) {
        // 检查文件类型
        if (!file.type.startsWith('image/')) {
            showError('请上传图片文件');
            return;
        }

        // 检查文件大小（限制为5MB）
        if (file.size > 5 * 1024 * 1024) {
            showError('图片大小不能超过5MB');
            return;
        }

        // 显示预览
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.style.display = 'block';
            previewImage.src = e.target.result;
            result.innerHTML = ''; // 清除之前的结果
        };
        reader.readAsDataURL(file);

        // 发送预测请求
        uploadAndPredict(file);
    }

    // 上传并预测函数
    function uploadAndPredict(file) {
        const formData = new FormData();
        formData.append('file', file);

        // 显示加载动画
        loading.style.display = 'flex';
        result.innerHTML = '';

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // 隐藏加载动画
            loading.style.display = 'none';

            if (data.success) {
                showSuccess(data);
            } else {
                showError(data.error);
            }
        })
        .catch(error => {
            loading.style.display = 'none';
            showError('请求失败：' + error.message);
        });
    }

    // 显示成功结果
    function showSuccess(data) {
        result.innerHTML = `
            <div class="success-result">
                <div class="result-icon">✓</div>
                <h3>预测结果</h3>
                <div class="result-content">
                    <p class="organ-name">${data.class_name}</p>
                    <div class="confidence-bar">
                        <div class="confidence-level" style="width: 100%"></div>
                    </div>
                </div>
            </div>
        `;
    }

    // 显示错误信息
    function showError(message) {
        result.innerHTML = `
            <div class="error-result">
                <div class="error-icon">!</div>
                <p class="error-message">${message}</p>
            </div>
        `;
    }

    // 处理示例图片点击
    document.querySelectorAll('.example-item img').forEach(img => {
        img.addEventListener('click', function() {
            // 从示例图片URL创建File对象
            fetch(this.src)
                .then(res => res.blob())
                .then(blob => {
                    const file = new File([blob], 'example.png', { type: 'image/png' });
                    handleFile(file);
                })
                .catch(error => {
                    showError('加载示例图片失败');
                });
        });
    });

    // 添加键盘支持
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            // ESC键清除结果
            preview.style.display = 'none';
            result.innerHTML = '';
            previewImage.src = '';
        }
    });

    // 添加移动设备触摸支持
    if ('ontouchstart' in window) {
        dropZone.addEventListener('touchstart', function(e) {
            e.preventDefault();
            fileInput.click();
        });
    }

    // 处理窗口粘贴事件
    document.addEventListener('paste', function(e) {
        const items = e.clipboardData.items;
        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') !== -1) {
                const file = items[i].getAsFile();
                handleFile(file);
                break;
            }
        }
    });

    // 添加页面可见性变化处理
    document.addEventListener('visibilitychange', function() {
        if (document.hidden) {
            // 页面隐藏时暂停任何进行中的操作
            if (loading.style.display === 'flex') {
                loading.style.display = 'none';
            }
        }
    });

    // 添加网络状态检测
    window.addEventListener('online', function() {
        document.body.classList.remove('offline');
    });

    window.addEventListener('offline', function() {
        document.body.classList.add('offline');
        showError('网络连接已断开');
    });

    // 添加页面卸载处理
    window.addEventListener('beforeunload', function(e) {
        if (loading.style.display === 'flex') {
            e.preventDefault();
            e.returnValue = '图片正在处理中，确定要离开吗？';
        }
    });
});