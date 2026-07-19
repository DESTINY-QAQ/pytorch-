(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  const modeInfo = {
    "cnn:catdog": {
      code: "MODULE / CNN / CATDOG",
      title: "卷积神经网络实验室",
      description: "上传真实图片运行 CatDogCNN，逐层追踪从像素到猫狗概率的数据流。",
      badge: "真实推理",
      real: true,
    },
    "cnn:resnet": {
      code: "MODULE / CNN / RESNET50",
      title: "ResNet50 残差网络实验室",
      description: "打开或关闭残差捷径，观察 F(x)+x 如何保护深层网络的信息通路。",
      badge: "教学模拟",
      real: false,
    },
    "transformer:vit": {
      code: "MODULE / TRANSFORMER / VIT-B16",
      title: "ViT-B/16 全局视觉实验室",
      description: "把图像切成 Patch，切换注意力头，观察不同区域如何交换全局线索。",
      badge: "教学模拟",
      real: false,
    },
    "transformer:translation": {
      code: "MODULE / TRANSFORMER / TRANSLATION",
      title: "Encoder–Decoder 翻译实验室",
      description: "从分词到交叉注意力，逐词模拟自回归翻译的完整工作流程。",
      badge: "教学模拟",
      real: false,
    },
    "yolo:detect": {
      code: "MODULE / YOLO11 / DETECTION",
      title: "YOLO11 实时检测实验室",
      description: "调节置信度、IoU、NMS 和模型规格，观察检测框与性能的实时变化。",
      badge: "教学模拟",
      real: false,
    },
  };

  const stepDetails = [
    ["预处理", "把任意尺寸的图片统一为模型认识的格式，就像先把不同试卷裁成统一大小。", "normalization"],
    ["卷积块 1", "第一层像找轮廓的探测器，扫描边缘、明暗交界等简单线索。", "convolution"],
    ["卷积块 2", "组合初级边缘形成毛发和纹理；池化保留重点并缩小地图。", "pooling"],
    ["卷积块 3", "更深的特征图开始响应耳朵、眼睛等较复杂的局部结构。", "feature-map"],
    ["卷积块 4", "把局部线索组合为更接近整体轮廓的高级语义特征。", "relu"],
    ["全局汇聚", "把每个通道的一整张特征图压成一个代表值，减少参数并适配分类器。", "adaptive-pooling"],
    ["分类器", "全连接层综合 128 条线索；Dropout 只在训练时随机遮住部分神经元。", "dropout"],
    ["概率输出", "Softmax 把两个分数转换成总和为 1 的概率，并选出概率更高的类别。", "softmax"],
  ];

  const glossary = {
    normalization: {
      title: "归一化 Normalize",
      definition: "按训练时使用的均值和标准差缩放像素，让不同图片进入相近的数值范围。",
      analogy: "像把来自不同学校、满分不同的成绩换算到同一把尺子上。",
      role: "CatDogCNN 的输入先缩放为 150×150，再转为张量并归一化，必须与训练阶段保持一致。",
      related: ["convolution", "feature-map"], target: ["cnn", "catdog", "normalization"],
    },
    convolution: {
      title: "卷积 Convolution",
      definition: "一个小型可学习滤镜在图像上滑动，对局部像素做加权计算，产生新的特征图。",
      analogy: "像侦探拿着小放大镜扫过现场，每次只看一小块，却用同一套判断标准寻找线索。",
      role: "CNN 用四组卷积逐层从边缘、纹理提炼到耳朵、脸型等高级特征；ResNet 的主干也由卷积组成。",
      related: ["feature-map", "relu", "pooling"], target: ["cnn", "catdog", "convolution"],
    },
    relu: {
      title: "ReLU 激活函数",
      definition: "把负数变成 0，正数原样保留，为网络加入非线性表达能力。",
      analogy: "像只允许有用的正信号通过的单向闸门。",
      role: "每次卷积后激活突出有效响应；没有非线性，再多层也只能近似一次线性变换。",
      related: ["convolution", "feature-map"], target: ["cnn", "catdog", "relu"],
    },
    pooling: {
      title: "最大池化 MaxPool",
      definition: "在小窗口中保留最大响应，缩小特征图，同时保留最明显的线索。",
      analogy: "把一段会议记录压缩成最关键的一句话。",
      role: "CatDogCNN 每组卷积后将宽高约减半，从 150 逐步变成 75、37、18、9。",
      related: ["convolution", "adaptive-pooling"], target: ["cnn", "catdog", "pooling"],
    },
    "adaptive-pooling": {
      title: "自适应平均池化",
      definition: "无论输入特征图多大，都按目标尺寸进行分区平均，输出固定大小。",
      analogy: "不管一篇文章多长，都为每个主题写出一个摘要数字。",
      role: "CatDogCNN 把 128×9×9 压成 128×1×1；ResNet50 也用全局平均池化连接分类器。",
      related: ["pooling", "classification-head"], target: ["cnn", "catdog", "adaptive-pooling"],
    },
    "feature-map": {
      title: "特征图 Feature Map",
      definition: "卷积核对某类视觉线索的响应分布；亮的区域表示该线索出现得更强。",
      analogy: "像热成像图，不显示原始照片，而显示某种线索在哪里最明显。",
      role: "通道数从 3 增至 32、64、128，让网络同时记录越来越多种视觉线索。",
      related: ["convolution", "pooling"], target: ["cnn", "catdog", "feature-map"],
    },
    dropout: {
      title: "Dropout",
      definition: "训练时随机把一部分神经元输出置零，迫使网络不要过度依赖少数路径。",
      analogy: "训练球队时随机让主力休息，避免整支队只会依靠一个明星。",
      role: "CatDogCNN 分类器在训练时使用 Dropout；真实预测时 model.eval() 会关闭它。",
      related: ["classification-head", "softmax"], target: ["cnn", "catdog", "dropout"],
    },
    softmax: {
      title: "Softmax",
      definition: "把任意实数分数转换为 0 到 1 的概率，并让所有类别概率之和等于 1。",
      analogy: "把候选人的原始得分换算为一张总计 100% 的支持率表。",
      role: "真实推理最后将 cats 与 dogs 的 logits 转为概率，最大值就是当前置信度。",
      related: ["confidence", "classification-head"], target: ["cnn", "catdog", "softmax"],
    },
    residual: {
      title: "残差连接 Residual Connection",
      definition: "把模块输入 x 绕过若干层，直接与模块计算结果 F(x) 相加，得到 F(x)+x。",
      analogy: "像给拥堵的城市道路增加一条直达高速，重要信息不必层层绕行。",
      role: "ResNet50 用捷径改善信息与梯度传播，使很深的网络更容易优化。",
      related: ["convolution", "adaptive-pooling"], target: ["cnn", "resnet", "residual"],
    },
    embedding: {
      title: "Embedding 嵌入",
      definition: "把离散 Token 或图像 Patch 映射成连续向量，使模型能用数字表达其含义。",
      analogy: "给每个词或图片块制作一张包含许多属性的数字身份证。",
      role: "ViT 将每个 16×16 Patch 映射为 768 维向量；翻译模型则把词 Token 映射为语义向量。",
      related: ["token", "positional-encoding"], target: ["transformer", "vit", "embedding"],
    },
    token: {
      title: "Token",
      definition: "模型处理序列时使用的基本单位，可以是一个字、词、子词或特殊符号。",
      analogy: "像用积木搭句子；模型一次观察和移动的是一块积木。",
      role: "翻译示例先把中文句子切成 Token，再逐个转换成 Embedding。",
      related: ["embedding", "autoregressive"], target: ["transformer", "translation", "token"],
    },
    "positional-encoding": {
      title: "位置编码 Positional Encoding",
      definition: "向每个 Token 的向量加入位置信息，让 Transformer 知道顺序或空间位置。",
      analogy: "给排队的人发号码牌，否则只看人群就不知道谁先谁后。",
      role: "ViT 用它记录 Patch 在图像中的位置；翻译模型用它区分词序。",
      related: ["embedding", "self-attention"], target: ["transformer", "vit", "positional-encoding"],
    },
    qkv: {
      title: "Q / K / V",
      definition: "注意力将每个输入投影成 Query（要找什么）、Key（我有什么）和 Value（我的内容）。",
      analogy: "Query 像搜索词，Key 像书的索引标签，Value 是书里的真正内容。",
      role: "Q 与 K 的相似度决定关注强弱，再按该权重汇总 V。",
      related: ["self-attention", "multi-head-attention"], target: ["transformer", "vit", "multi-head-attention"],
    },
    "self-attention": {
      title: "自注意力 Self-Attention",
      definition: "序列中每个位置都与同一序列的其他位置计算相关性，再有选择地汇总信息。",
      analogy: "开圆桌会议时，每个人都能根据当前问题选择重点听谁发言。",
      role: "Encoder 用它理解原句上下文；ViT 用它让远隔很远的 Patch 直接交换视觉线索。",
      related: ["qkv", "multi-head-attention", "cross-attention"], target: ["transformer", "translation", "self-attention"],
    },
    "multi-head-attention": {
      title: "多头注意力 Multi-Head Attention",
      definition: "并行使用多组 Q/K/V 投影，让不同注意力头学习不同类型的关系。",
      analogy: "让颜色、轮廓、位置等多位专家同时观察同一张图，最后合并报告。",
      role: "ViT-B/16 的每层 Encoder 有 12 个头；本实验可切换 Head 观察不同关注图案。",
      related: ["qkv", "self-attention"], target: ["transformer", "vit", "multi-head-attention"],
    },
    "masked-attention": {
      title: "遮罩注意力 Masked Attention",
      definition: "解码器只能关注已生成位置，未来 Token 被遮住，避免训练时偷看答案。",
      analogy: "做填空题时用纸遮住后面的标准答案，只能根据已经写出的内容继续。",
      role: "经典 Transformer Decoder 用因果遮罩确保输出可以从左到右逐词生成。",
      related: ["autoregressive", "cross-attention"], target: ["transformer", "translation", "masked-attention"],
    },
    "cross-attention": {
      title: "交叉注意力 Cross-Attention",
      definition: "Query 来自解码器，Key 和 Value 来自编码器，让输出序列查询输入序列。",
      analogy: "译者写英文时随时回看中文原稿，寻找当前词对应的原文线索。",
      role: "解码每个英文词时，模型用它从已编码的中文信息中选取相关内容。",
      related: ["self-attention", "qkv", "autoregressive"], target: ["transformer", "translation", "cross-attention"],
    },
    autoregressive: {
      title: "自回归生成",
      definition: "一次预测一个新 Token，并把已经生成的 Token 作为下一次预测的输入。",
      analogy: "像接龙：每说出一个词，下一步都要把前面完整内容一起考虑。",
      role: "点击“生成下一词”可观察英文译文按 <BOS> → 单词 → <EOS> 的顺序生长。",
      related: ["token", "masked-attention"], target: ["transformer", "translation", "autoregressive"],
    },
    backbone: {
      title: "Backbone 主干网络",
      definition: "负责从输入图片中提取分层视觉特征的主要网络部分。",
      analogy: "像侦察队先把现场的边缘、纹理、形状等证据完整收集回来。",
      role: "YOLO11 Backbone 产生不同深度和分辨率的特征，交给 Neck 继续融合。",
      related: ["neck", "detection-head"], target: ["yolo", "detect", "backbone"],
    },
    neck: {
      title: "Neck 特征融合层",
      definition: "连接 Backbone 与检测头，融合多个尺度的浅层细节和深层语义。",
      analogy: "像情报中心把近景高清照片和远景全局地图叠在一起分析。",
      role: "帮助 YOLO 同时识别画面中的大目标和很小的远处目标。",
      related: ["backbone", "detection-head"], target: ["yolo", "detect", "neck"],
    },
    "detection-head": {
      title: "Detect Head 检测头",
      definition: "把融合特征转换为候选框位置、类别分数和置信度。",
      analogy: "侦察和情报整理结束后，由报告员在地图上画框并贴上名称。",
      role: "YOLO 在多个尺度输出候选框，之后还要经过置信度筛选与 NMS。",
      related: ["confidence", "nms", "backbone"], target: ["yolo", "detect", "detection-head"],
    },
    "classification-head": {
      title: "分类头 Classification Head",
      definition: "把网络提取的综合特征映射为每个类别的最终分数。",
      analogy: "像汇总所有侦探报告后，最后给每个嫌疑人打一个总分。",
      role: "CatDogCNN 输出 2 个分数；ViT 用 CLS Token 的表示完成分类；ResNet50 用 FC 输出类别。",
      related: ["softmax", "adaptive-pooling"], target: ["cnn", "catdog", "dropout"],
    },
    confidence: {
      title: "置信度 Confidence",
      definition: "模型对某个候选类别或检测框的确信程度；阈值越高，保留结果通常越少。",
      analogy: "像设置入场考试分数线：分数线提高，能进入的候选人会变少。",
      role: "拖动 YOLO 置信度阈值会即时过滤低分候选框；CNN 则显示最高类别概率。",
      related: ["nms", "iou", "softmax"], target: ["yolo", "detect", "confidence"],
    },
    iou: {
      title: "IoU 交并比",
      definition: "两个框交集面积除以并集面积，用 0～1 衡量它们的重叠程度。",
      analogy: "把两张透明贴纸叠起来，重叠部分占两张贴纸合并面积的比例。",
      role: "NMS 用 IoU 判断两个同类候选框是否可能描述同一个目标。",
      related: ["nms", "confidence"], target: ["yolo", "detect", "nms"],
    },
    nms: {
      title: "NMS 非极大值抑制",
      definition: "先保留最高分框，再移除与它高度重叠的同类低分框，重复直到没有候选。",
      analogy: "同一个人被多名记者重复登记时，只保留最清楚、得分最高的一份记录。",
      role: "关闭 NMS 可看到虚线重复框；调整 IoU 阈值会改变哪些重叠框被移除。",
      related: ["iou", "confidence", "detection-head"], target: ["yolo", "detect", "nms"],
    },
  };

  const state = {
    model: "cnn",
    modes: { cnn: "catdog", transformer: "vit", yolo: "detect" },
    steps: { "cnn:catdog": 0, "cnn:resnet": 0, "transformer:vit": 0, "transformer:translation": 0, "yolo:detect": 0 },
    playing: false,
    timer: null,
    speed: 950,
    selectedPatch: 89,
    attentionHead: 0,
    lastFocus: null,
    currentTerm: null,
  };

  const currentKey = () => `${state.model}:${state.modes[state.model]}`;
  const activePanel = () => $(`[data-lab="${state.model}"][data-mode-panel="${state.modes[state.model]}"]`);
  const flowSteps = () => $$(".flow-step", activePanel());

  function pause() {
    state.playing = false;
    window.clearInterval(state.timer);
    state.timer = null;
    $("#play-toggle b").textContent = "播放流程";
    $("#play-toggle span").textContent = "▶";
  }

  function applyStep(index, options = {}) {
    const steps = flowSteps();
    if (!steps.length) return;
    const bounded = Math.max(0, Math.min(index, steps.length - 1));
    state.steps[currentKey()] = bounded;
    steps.forEach((step, i) => {
      step.classList.toggle("active", i === bounded);
      step.classList.toggle("complete", i < bounded);
    });
    $$(".flow-link", activePanel()).forEach((link, i) => link.classList.toggle("lit", i < bounded));
    $("#step-label").textContent = `步骤 ${bounded + 1} / ${steps.length}`;
    $("#timeline-progress").style.width = `${((bounded + 1) / steps.length) * 100}%`;

    if (currentKey() === "cnn:catdog") {
      const detail = stepDetails[bounded];
      $("#catdog-step-title").textContent = detail[0];
      $("#catdog-step-copy").textContent = detail[1];
      const termButton = $(".step-inspector button");
      termButton.dataset.term = detail[2];
    }
    if (options.scroll) steps[bounded].scrollIntoView({ behavior: "smooth", block: "center" });
  }

  function nextStep() {
    const steps = flowSteps();
    const current = state.steps[currentKey()] || 0;
    if (current >= steps.length - 1) {
      pause();
      return false;
    }
    applyStep(current + 1);
    return true;
  }

  function play() {
    if (state.playing) { pause(); return; }
    if ((state.steps[currentKey()] || 0) >= flowSteps().length - 1) applyStep(0);
    state.playing = true;
    $("#play-toggle b").textContent = "暂停流程";
    $("#play-toggle span").textContent = "Ⅱ";
    state.timer = window.setInterval(nextStep, state.speed);
  }

  function switchLab(model, mode = state.modes[model]) {
    pause();
    state.model = model;
    state.modes[model] = mode;
    $$(".model-tab").forEach(tab => {
      const active = tab.dataset.model === model;
      tab.classList.toggle("active", active);
      tab.setAttribute("aria-selected", String(active));
    });
    $$("[data-submodes]").forEach(group => { group.hidden = group.dataset.submodes !== model; });
    $$(`[data-submodes="${model}"] button`).forEach(button => button.classList.toggle("active", button.dataset.mode === mode));
    $$(".lab-panel").forEach(panel => {
      const active = panel.dataset.lab === model && panel.dataset.modePanel === mode;
      panel.hidden = !active;
      panel.classList.toggle("active", active);
      if (active) {
        panel.classList.remove("entering");
        void panel.offsetWidth;
        panel.classList.add("entering");
      }
    });
    const info = modeInfo[`${model}:${mode}`];
    $("#console-code").textContent = info.code;
    $("#console-title").textContent = info.title;
    $("#console-description").textContent = info.description;
    $("#mode-badge").className = `mode-badge ${info.real ? "real" : "sim"}`;
    $("#mode-badge").innerHTML = `<i></i> ${info.badge}`;
    applyStep(state.steps[currentKey()] || 0);
    if (currentKey() === "transformer:vit") requestAnimationFrame(drawAttention);
  }

  $$(".model-tab").forEach(tab => tab.addEventListener("click", () => switchLab(tab.dataset.model)));
  $$(".submodes button").forEach(button => button.addEventListener("click", () => switchLab(state.model, button.dataset.mode)));
  $("#play-toggle").addEventListener("click", play);
  $("#next-step").addEventListener("click", () => { pause(); nextStep(); });
  $("#reset-lab").addEventListener("click", () => { pause(); applyStep(0); resetModeExtras(); });
  $("#speed-select").addEventListener("change", event => {
    state.speed = Number(event.target.value);
    if (state.playing) { pause(); play(); }
  });
  document.addEventListener("click", event => {
    const step = event.target.closest(".flow-step");
    if (!step || !activePanel()?.contains(step)) return;
    const index = flowSteps().indexOf(step);
    if (index >= 0) { pause(); applyStep(index); }
  });

  // Real CatDogCNN prediction.
  const imageInput = $("#image-input");
  const dropZone = $("#drop-zone");
  const preview = $("#client-preview");
  const previewImage = $("#preview-image");
  const emptyState = $("#empty-state");
  const submitButton = $("#submit-button");
  const imageReset = $("#image-reset");
  const allowedTypes = ["image/jpeg", "image/png", "image/bmp", "image/webp"];

  function readableSize(bytes) {
    return bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1024 / 1024).toFixed(2)} MB`;
  }

  function setFormMessage(message = "", type = "") {
    $("#form-message").textContent = message;
    $("#form-message").className = `form-message ${type}`;
  }

  function showFile(file) {
    if (!file || !allowedTypes.includes(file.type) || file.size > 10 * 1024 * 1024) {
      imageInput.value = "";
      setFormMessage("请选择不超过 10 MB 的 JPG、PNG、BMP 或 WEBP 图片。", "error");
      return;
    }
    const reader = new FileReader();
    reader.onload = event => {
      previewImage.src = event.target.result;
      $("#file-name").textContent = file.name;
      $("#file-size").textContent = readableSize(file.size);
      emptyState.hidden = true;
      preview.hidden = false;
      submitButton.disabled = false;
      imageReset.disabled = false;
      setFormMessage("图片已装载，可以启动真实推理。", "success");
      switchLab("cnn", "catdog");
      applyStep(0);
    };
    reader.readAsDataURL(file);
  }

  function clearImage() {
    imageInput.value = "";
    previewImage.removeAttribute("src");
    preview.hidden = true;
    emptyState.hidden = false;
    submitButton.disabled = true;
    imageReset.disabled = true;
    preview.classList.remove("scanning");
    setFormMessage();
    $("#prediction-core").className = "prediction-core";
    $("#result-label").textContent = "待机";
    $("#result-confidence").textContent = "--";
    $("#result-class").textContent = "上传图片后显示结果";
    $("#confidence-text").textContent = "0%";
    $("#confidence-bar").style.width = "0%";
  }

  imageInput.addEventListener("change", () => showFile(imageInput.files[0]));
  ["dragenter", "dragover"].forEach(name => dropZone.addEventListener(name, event => { event.preventDefault(); dropZone.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach(name => dropZone.addEventListener(name, event => { event.preventDefault(); dropZone.classList.remove("dragging"); }));
  dropZone.addEventListener("drop", event => {
    const file = event.dataTransfer.files[0];
    if (!file) return;
    try {
      const transfer = new DataTransfer();
      transfer.items.add(file);
      imageInput.files = transfer.files;
    } catch (_error) { /* Browsers may protect FileList; the FormData fallback below still uses input when available. */ }
    showFile(file);
  });
  imageReset.addEventListener("click", clearImage);

  $("#predict-form").addEventListener("submit", async event => {
    event.preventDefault();
    const file = imageInput.files[0];
    if (!file) { setFormMessage("请先选择图片。", "error"); return; }
    const formData = new FormData();
    formData.append("image", file, file.name);
    submitButton.disabled = true;
    imageReset.disabled = true;
    submitButton.classList.add("loading");
    preview.classList.add("scanning");
    $("#prediction-core").classList.add("active");
    setFormMessage("正在执行 PyTorch CPU 推理…");
    try {
      const response = await fetch("/api/cnn/predict", { method: "POST", body: formData });
      const payload = await response.json().catch(() => ({ ok: false, error: "服务返回了无法解析的响应。" }));
      if (!response.ok || !payload.ok) throw new Error(payload.error || "推理失败，请稍后重试。");
      const percent = Math.round(payload.prediction.confidence * 10000) / 100;
      previewImage.src = payload.preview;
      $("#prediction-core").classList.add("has-result");
      $("#result-label").textContent = payload.prediction.label;
      $("#result-confidence").textContent = `${percent}%`;
      $("#result-class").textContent = `${payload.prediction.id} · ${payload.model.runtime}`;
      $("#confidence-text").textContent = `${percent}%`;
      $("#confidence-bar").style.width = `${percent}%`;
      setFormMessage(`真实推理完成：模型判断为“${payload.prediction.label}”。`, "success");
      applyStep(0);
      play();
    } catch (error) {
      $("#prediction-core").classList.remove("active");
      setFormMessage(error.message, "error");
    } finally {
      submitButton.classList.remove("loading");
      preview.classList.remove("scanning");
      submitButton.disabled = false;
      imageReset.disabled = false;
    }
  });

  // ResNet residual connection comparison.
  function updateResidual() {
    const enabled = $("#residual-toggle").checked;
    $(".resnet-track").classList.toggle("no-residual", !enabled);
    $("#residual-demo").classList.toggle("disabled", !enabled);
    $("#residual-score").textContent = enabled ? "92%" : "61%";
    $(".signal-meter i").style.width = enabled ? "92%" : "61%";
    $("#residual-copy").textContent = enabled
      ? "捷径为信息和梯度提供更直接的通路，让更深的网络更容易训练。"
      : "关闭捷径后，信号必须连续穿过所有变换层；这里用衰减动画演示深层传播更困难。";
  }
  $("#residual-toggle").addEventListener("change", updateResidual);

  // ViT patch matrix and deterministic attention field.
  function patchColor(index) {
    const row = Math.floor(index / 14);
    const col = index % 14;
    const center = Math.hypot(row - 6.5, col - 6.5);
    const hue = 195 + Math.round((col / 13) * 45);
    const light = 18 + Math.round((1 - Math.min(center / 10, 1)) * 28) + ((row * 7 + col * 3) % 8);
    return `hsl(${hue} 58% ${light}%)`;
  }

  function buildPatchGrid() {
    const grid = $("#patch-grid");
    for (let i = 0; i < 196; i += 1) {
      const button = document.createElement("button");
      button.type = "button";
      button.style.setProperty("--patch-color", patchColor(i));
      button.setAttribute("aria-label", `Patch ${String(i).padStart(3, "0")}`);
      button.classList.toggle("selected", i === state.selectedPatch);
      button.addEventListener("click", () => selectPatch(i));
      grid.appendChild(button);
    }
    const heads = $("#head-selector");
    for (let i = 0; i < 12; i += 1) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = String(i + 1).padStart(2, "0");
      button.title = `Attention Head ${i + 1}`;
      button.classList.toggle("active", i === state.attentionHead);
      button.addEventListener("click", () => {
        state.attentionHead = i;
        $$("button", heads).forEach((item, index) => item.classList.toggle("active", index === i));
        $("#selected-head").textContent = `Head ${i + 1}`;
        drawAttention();
      });
      heads.appendChild(button);
    }
  }

  function selectPatch(index) {
    state.selectedPatch = index;
    $$("#patch-grid button").forEach((button, i) => button.classList.toggle("selected", i === index));
    $("#selected-patch").textContent = `P${String(index).padStart(3, "0")}`;
    drawAttention();
  }

  function attentionWeight(from, to, head) {
    const fr = Math.floor(from / 14), fc = from % 14, tr = Math.floor(to / 14), tc = to % 14;
    const distance = Math.hypot(fr - tr, fc - tc);
    const directional = Math.abs(Math.sin((tr * (head + 2) + tc * (head + 5) + from) * .37));
    const local = Math.exp(-distance / (2.1 + (head % 4)));
    return Math.min(1, local * .64 + directional * .48);
  }

  function drawAttention() {
    const canvas = $("#attention-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    const marginX = 52, marginY = 24, stepX = (w - marginX * 2) / 13, stepY = (h - marginY * 2) / 13;
    const point = index => ({ x: marginX + (index % 14) * stepX, y: marginY + Math.floor(index / 14) * stepY });
    const source = point(state.selectedPatch);
    const ranked = Array.from({ length: 196 }, (_, i) => ({ i, weight: attentionWeight(state.selectedPatch, i, state.attentionHead) }))
      .filter(item => item.i !== state.selectedPatch).sort((a, b) => b.weight - a.weight).slice(0, 22);
    ranked.reverse().forEach(item => {
      const target = point(item.i);
      const gradient = ctx.createLinearGradient(source.x, source.y, target.x, target.y);
      gradient.addColorStop(0, `rgba(56,217,255,${.1 + item.weight * .55})`);
      gradient.addColorStop(1, `rgba(160,108,255,${.04 + item.weight * .35})`);
      ctx.beginPath(); ctx.moveTo(source.x, source.y);
      const bend = ((item.i + state.attentionHead) % 2 ? 1 : -1) * 18;
      ctx.quadraticCurveTo((source.x + target.x) / 2 + bend, (source.y + target.y) / 2 - bend, target.x, target.y);
      ctx.strokeStyle = gradient; ctx.lineWidth = .5 + item.weight * 2.1; ctx.stroke();
    });
    for (let i = 0; i < 196; i += 1) {
      const p = point(i); const weight = attentionWeight(state.selectedPatch, i, state.attentionHead);
      ctx.beginPath(); ctx.arc(p.x, p.y, i === state.selectedPatch ? 5.5 : 1.4 + weight * 1.5, 0, Math.PI * 2);
      ctx.fillStyle = i === state.selectedPatch ? "#ffffff" : `rgba(56,217,255,${.15 + weight * .75})`;
      ctx.shadowColor = i === state.selectedPatch ? "#38d9ff" : "transparent"; ctx.shadowBlur = i === state.selectedPatch ? 16 : 0; ctx.fill();
    }
    ctx.shadowBlur = 0;
  }

  // Deterministic translation demonstration.
  const translationExamples = [
    { source: ["机器", "学习", "很", "有趣", "。"], target: ["Machine", "learning", "is", "interesting", ".", "<EOS>"] },
    { source: ["猫", "坐", "在", "窗边", "。"], target: ["The", "cat", "sits", "by", "the", "window", ".", "<EOS>"] },
    { source: ["我们", "一起", "探索", "未来", "。"], target: ["We", "explore", "the", "future", "together", ".", "<EOS>"] },
  ];
  let translationIndex = 0;
  let generatedCount = 0;
  let selectedSourceToken = 0;

  function renderTranslation() {
    const example = translationExamples[translationIndex];
    const source = $("#source-tokens");
    source.innerHTML = "";
    example.source.forEach((token, index) => {
      const button = document.createElement("button");
      button.type = "button"; button.className = "token-chip"; button.textContent = token;
      button.classList.toggle("active", index === selectedSourceToken);
      button.addEventListener("click", () => { selectedSourceToken = index; renderTranslation(); });
      source.appendChild(button);
    });
    const target = $("#target-tokens");
    target.innerHTML = '<span class="token-chip generated">&lt;BOS&gt;</span>';
    example.target.slice(0, generatedCount).forEach(token => {
      const span = document.createElement("span"); span.className = "token-chip generated"; span.textContent = token; target.appendChild(span);
    });
    const complete = generatedCount >= example.target.length;
    $("#next-token").disabled = complete;
    $("#generation-status").textContent = complete ? "生成完成：<EOS> 表示句子结束。" : `已生成 ${generatedCount} / ${example.target.length} 个 Token。模型每次只能看到已经生成的部分。`;
    renderHeatmap(example);
  }

  function renderHeatmap(example) {
    const heatmap = $("#attention-heatmap");
    const rows = Math.max(1, generatedCount || 1);
    heatmap.style.gridTemplateColumns = `repeat(${example.source.length}, 1fr)`;
    heatmap.innerHTML = "";
    for (let row = 0; row < rows; row += 1) {
      example.source.forEach((token, col) => {
        const cell = document.createElement("span");
        const focus = 1 - Math.min(Math.abs(col - ((row + translationIndex) % example.source.length)) / example.source.length, 1);
        const selectedBoost = col === selectedSourceToken ? .25 : 0;
        cell.className = "heat-cell";
        cell.style.setProperty("--heat", String(Math.min(.92, .12 + focus * .55 + selectedBoost)));
        cell.title = `${example.target[Math.min(row, example.target.length - 1)] || "起始"} → ${token}`;
        heatmap.appendChild(cell);
      });
    }
  }

  $("#sentence-select").addEventListener("change", event => {
    translationIndex = Number(event.target.value); generatedCount = 0; selectedSourceToken = 0; renderTranslation();
  });
  $("#next-token").addEventListener("click", () => {
    const example = translationExamples[translationIndex];
    generatedCount = Math.min(generatedCount + 1, example.target.length);
    const steps = flowSteps();
    if (currentKey() === "transformer:translation" && steps.length) applyStep(Math.min(steps.length - 1, 3 + Math.floor(generatedCount / 2)));
    renderTranslation();
  });

  // YOLO11 deterministic detection simulation.
  const detections = [
    { id: "car-main", label: "car", score: .94, x: 35, y: 57, w: 30, h: 23, color: "#38d9ff" },
    { id: "person-a", label: "person", score: .88, x: 16, y: 51, w: 10, h: 33, color: "#55ecac" },
    { id: "person-b", label: "person", score: .81, x: 75, y: 43, w: 10, h: 39, color: "#55ecac" },
    { id: "dog-main", label: "dog", score: .76, x: 84, y: 72, w: 13, h: 19, color: "#ff9b46" },
    { id: "bike-main", label: "bicycle", score: .61, x: 6, y: 71, w: 22, h: 22, color: "#a06cff" },
    { id: "car-dup", label: "car", score: .64, x: 37, y: 59, w: 28, h: 21, color: "#38d9ff", duplicate: true },
    { id: "person-dup", label: "person", score: .46, x: 74, y: 45, w: 12, h: 37, color: "#55ecac", duplicate: true },
    { id: "traffic-light", label: "traffic light", score: .24, x: 65, y: 27, w: 5, h: 16, color: "#ff637d" },
  ];
  const yoloSpecs = {
    n: { latency: 3.4, accuracy: 71.8 }, s: { latency: 4.8, accuracy: 76.9 }, m: { latency: 7.2, accuracy: 82.1 },
    l: { latency: 10.6, accuracy: 84.3 }, x: { latency: 15.8, accuracy: 85.7 },
  };

  function boxIoU(a, b) {
    const left = Math.max(a.x, b.x), top = Math.max(a.y, b.y), right = Math.min(a.x + a.w, b.x + b.w), bottom = Math.min(a.y + a.h, b.y + b.h);
    const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
    return intersection / (a.w * a.h + b.w * b.h - intersection || 1);
  }

  function nmsFilter(candidates, threshold) {
    const sorted = [...candidates].sort((a, b) => b.score - a.score);
    const kept = [];
    sorted.forEach(candidate => {
      if (!kept.some(item => item.label === candidate.label && boxIoU(item, candidate) > threshold)) kept.push(candidate);
    });
    return kept;
  }

  function updateYolo() {
    const confidence = Number($("#conf-slider").value);
    const iou = Number($("#iou-slider").value);
    const nmsOn = $("#nms-toggle").checked;
    $("#conf-output").textContent = confidence.toFixed(2);
    $("#iou-output").textContent = iou.toFixed(2);
    let visible = detections.filter(item => item.score >= confidence);
    if (nmsOn) visible = nmsFilter(visible, iou);
    const container = $("#detection-boxes"); container.innerHTML = "";
    visible.forEach((item, index) => {
      const box = document.createElement("div");
      box.className = `detect-box${item.duplicate ? " duplicate" : ""}`;
      box.style.cssText = `left:${item.x}%;top:${item.y}%;width:${item.w}%;height:${item.h}%;--box-color:${item.color};animation-delay:${index * 35}ms`;
      box.innerHTML = `<span>${item.label} ${(item.score * 100).toFixed(0)}%</span>`;
      container.appendChild(box);
    });
    const spec = yoloSpecs[$("#yolo-size").value];
    $("#box-count").textContent = String(visible.length);
    $("#latency").textContent = `${spec.latency.toFixed(1)} ms`;
    $("#accuracy").textContent = spec.accuracy.toFixed(1);
  }
  ["#conf-slider", "#iou-slider", "#nms-toggle", "#yolo-size"].forEach(selector => $(selector).addEventListener("input", updateYolo));

  // Glossary drawer, deep links, related terms and workflow jumps.
  const drawer = $("#knowledge-drawer");
  const backdrop = $("#drawer-backdrop");

  function openTerm(slug, updateHash = true) {
    const term = glossary[slug];
    if (!term) return;
    state.currentTerm = slug;
    state.lastFocus = document.activeElement;
    $("#term-title").textContent = term.title;
    $("#term-definition").textContent = term.definition;
    $("#term-analogy").textContent = term.analogy;
    $("#term-role").textContent = term.role;
    const related = $("#related-terms"); related.innerHTML = "";
    term.related.forEach(key => {
      if (!glossary[key]) return;
      const button = document.createElement("button"); button.type = "button"; button.textContent = glossary[key].title;
      button.addEventListener("click", () => openTerm(key)); related.appendChild(button);
    });
    drawer.classList.add("open"); drawer.setAttribute("aria-hidden", "false"); backdrop.hidden = false;
    document.body.style.overflow = "hidden";
    $("#drawer-close").focus();
    if (updateHash && location.hash !== `#term=${slug}`) location.hash = `term=${slug}`;
  }

  function closeTerm(clearHash = true) {
    drawer.classList.remove("open"); drawer.setAttribute("aria-hidden", "true"); backdrop.hidden = true;
    document.body.style.overflow = "";
    if (clearHash && location.hash.startsWith("#term=")) history.pushState(null, "", `${location.pathname}${location.search}`);
    if (state.lastFocus && typeof state.lastFocus.focus === "function") state.lastFocus.focus();
  }

  document.addEventListener("click", event => {
    const trigger = event.target.closest("[data-term]");
    if (!trigger) return;
    event.preventDefault(); event.stopPropagation(); openTerm(trigger.dataset.term);
  });
  $("#drawer-close").addEventListener("click", () => closeTerm());
  backdrop.addEventListener("click", () => closeTerm());
  document.addEventListener("keydown", event => {
    if (event.key === "Escape" && drawer.classList.contains("open")) closeTerm();
    if (event.key === "Tab" && drawer.classList.contains("open")) {
      const focusable = $$('button,[href],select,input,[tabindex]:not([tabindex="-1"])', drawer).filter(element => !element.disabled);
      const first = focusable[0], last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }
  });
  window.addEventListener("hashchange", () => {
    const match = location.hash.match(/^#term=([\w-]+)$/);
    if (match && glossary[match[1]]) openTerm(match[1], false);
    else if (drawer.classList.contains("open")) closeTerm(false);
  });
  $("#jump-to-step").addEventListener("click", () => {
    const term = glossary[state.currentTerm];
    if (!term?.target) return;
    const [model, mode, termSlug] = term.target;
    closeTerm(); switchLab(model, mode);
    const steps = flowSteps();
    const target = steps.find(step => step.dataset.term === termSlug || step.dataset.term === state.currentTerm);
    if (target) applyStep(steps.indexOf(target), { scroll: true });
    else activePanel().scrollIntoView({ behavior: "smooth", block: "center" });
  });

  function showToast(message) {
    const toast = $("#toast"); toast.textContent = message; toast.classList.add("show");
    window.clearTimeout(showToast.timer); showToast.timer = window.setTimeout(() => toast.classList.remove("show"), 1800);
  }
  $("#copy-term-link").addEventListener("click", async () => {
    const url = `${location.origin}${location.pathname}#term=${state.currentTerm}`;
    try { await navigator.clipboard.writeText(url); showToast("术语链接已复制"); }
    catch (_error) {
      const area = document.createElement("textarea"); area.value = url; document.body.appendChild(area); area.select(); document.execCommand("copy"); area.remove(); showToast("术语链接已复制");
    }
  });

  function resetModeExtras() {
    if (currentKey() === "cnn:resnet") { $("#residual-toggle").checked = true; updateResidual(); }
    if (currentKey() === "transformer:vit") { state.selectedPatch = 89; state.attentionHead = 0; selectPatch(89); $$("#head-selector button").forEach((b, i) => b.classList.toggle("active", i === 0)); $("#selected-head").textContent = "Head 1"; }
    if (currentKey() === "transformer:translation") { generatedCount = 0; selectedSourceToken = 0; renderTranslation(); }
    if (currentKey() === "yolo:detect") { $("#conf-slider").value = .35; $("#iou-slider").value = .5; $("#nms-toggle").checked = true; $("#yolo-size").value = "m"; updateYolo(); }
  }

  // Lightweight offline star field; disabled when the OS asks for reduced motion.
  function startStarfield() {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const canvas = $("#starfield"), ctx = canvas.getContext("2d");
    let stars = [], raf = 0;
    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.floor(innerWidth * dpr); canvas.height = Math.floor(innerHeight * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      stars = Array.from({ length: Math.min(130, Math.floor(innerWidth / 9)) }, (_, i) => ({
        x: (i * 83.17) % innerWidth, y: (i * 47.41) % innerHeight, r: .3 + (i % 5) * .15, speed: .04 + (i % 7) * .018,
      }));
    }
    function draw() {
      ctx.clearRect(0, 0, innerWidth, innerHeight);
      stars.forEach(star => {
        star.y += star.speed; if (star.y > innerHeight) star.y = 0;
        ctx.beginPath(); ctx.arc(star.x, star.y, star.r, 0, Math.PI * 2); ctx.fillStyle = "rgba(117,210,255,.58)"; ctx.fill();
      });
      raf = requestAnimationFrame(draw);
    }
    resize(); draw();
    window.addEventListener("resize", () => { cancelAnimationFrame(raf); resize(); draw(); }, { passive: true });
  }

  buildPatchGrid();
  renderTranslation();
  updateYolo();
  updateResidual();
  switchLab("cnn", "catdog");
  startStarfield();
  const initialTerm = location.hash.match(/^#term=([\w-]+)$/);
  if (initialTerm && glossary[initialTerm[1]]) window.setTimeout(() => openTerm(initialTerm[1], false), 0);
})();
