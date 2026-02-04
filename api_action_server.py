####深度当前问题现在取前7张，结合当前的刚好9张可以一批次进行处理
import rclpy
import time
import sys
import os
import base64
import sqlite3
import ast
import threading
import json
import warnings
import numpy as np
import torch
import clip
from PIL import Image
from datetime import datetime, timedelta
from dateutil import parser
from typing import List, Optional, Dict, Any, Set, Tuple
from concurrent.futures import ThreadPoolExecutor
import queue

# ROS2相关
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.client import Client
# ROS2标准服务接口
from std_srvs.srv import Trigger
# 导入编译后的 Action 接口
from api_action.action import ApiAction

# 导入通义千问
try:
    import dashscope
    from dashscope import MultiModalConversation
    print("dashscope 包导入成功", file=sys.stderr)
except ImportError as e:
    print(f"错误: 缺少dashscope包，请执行：pip install dashscope --user --break-system-packages", file=sys.stderr)
    sys.exit(1)

# 关闭无关警告，工控机终端整洁
warnings.filterwarnings('ignore')

# ====================== 基础配置（统一表名：仅保留detection_objects）=====================
# 大模型配置
MODEL_NAME = "qwen-vl-max"
TEMPERATURE = 0.0
MAX_TOKENS_CLASSIFY = 64
MAX_TOKENS_INFER = 2048
MAX_TOKENS_FUSION = 3072
MAX_TOKENS_CHAT = 512
MAX_TOKENS_CURRENT_SCENE = 2048  # 简单当前场景（CLIP）专用
MAX_TOKENS_DEPTH_CURRENT = 3072  # 深度当前场景（8张图单批次）专用

# 支持意图（6类）
SUPPORTED_INTENTS = ["视觉理解", "简单目标检索", "深度目标检索",
                     "简单当前场景问题", "深度当前场景问题", "闲聊"]
# 深度操作反馈间隔（秒）
DEPTH_RETRIEVAL_FEEDBACK_INTERVAL = 0.5

# CLIP核显加速+匹配配置（可直接修改）
CONFIDENCE_THRESHOLD = 0.55  # CLIP匹配置信度阈值78%
TOP_K = 8                    # 取前TOP8匹配结果
DB_FILE_PATH = "/home/robot-5001/ros2_ws/src/a_vision_memory_qa_node/a_vision_memory_qa_node/detection.db"  # 统一DB路径
DB_TABLE_NAME = "detection_objects"  # 所有逻辑统一使用该表
MIN_DETECTION_CONFIDENCE = 0.5 # 全局最小检测置信度，过滤低置信数据
TIME_WINDOW_HOURS = 72
# 🔥 核心固定：深度当前场景 1张当前+7张历史 = 8张，取消原MAX_IMG_PER_REQUEST
FIXED_TOTAL_IMG = 8    # 总张数
FIXED_CURRENT_IMG = 1  # 固定当前图1张
FIXED_HISTORY_IMG = 7  # 固定历史最新7张

# COCO映射：人和物统一（所有场景共用）
COCO80_CN2EN = {
    "人": "person",
    "自行车": "bicycle", "汽车": "car", "摩托车": "motorcycle", "飞机": "airplane", "公交车": "bus",
    "火车": "train", "卡车": "truck", "船": "boat", "红绿灯": "traffic light", "消防栓": "fire hydrant",
    "停车牌": "stop sign", "停车计时器": "parking meter", "长凳": "bench",
    "鸟": "bird", "猫": "cat", "狗": "dog", "马": "horse", "羊": "sheep",
    "牛": "cow", "大象": "elephant", "熊": "bear", "斑马": "zebra", "长颈鹿": "giraffe",
    "背包": "backpack", "雨伞": "umbrella", "包": "handbag", "领带": "tie", "行李箱": "suitcase",
    "飞盘": "frisbee", "滑雪板": "skis", "冲浪板": "surfboard", "网球拍": "tennis racket", "瓶子": "bottle",
    "酒杯": "wine glass", "杯子": "cup", "叉子": "fork", "刀": "knife", "勺子": "spoon", "碗": "bowl",
    "香蕉": "banana", "苹果": "apple", "三明治": "sandwich", "橙子": "orange", "西兰花": "broccoli",
    "胡萝卜": "carrot", "热狗": "hot dog", "披萨": "pizza", "甜甜圈": "donut", "蛋糕": "cake",
    "椅子": "chair", "沙发": "couch", "盆栽": "potted plant", "床": "bed", "餐桌": "dining table",
    "马桶": "toilet", "电视": "tv", "笔记本电脑": "laptop", "鼠标": "mouse", "遥控器": "remote",
    "键盘": "keyboard", "手机": "cell phone", "微波炉": "microwave", "烤箱": "oven", "烤面包机": "toaster",
    "水槽": "sink", "冰箱": "refrigerator",
    "书": "book", "时钟": "clock", "花瓶": "vase", "剪刀": "scissors", "泰迪熊": "teddy bear",
    "吹风机": "hair drier", "牙刷": "toothbrush"
}
COCO80_EN2CN = {v: k for k, v in COCO80_CN2EN.items()}
SUPPORTED_TARGETS = list(COCO80_CN2EN.keys())

# ====================== 英特尔核显满血加速配置（CLIP专用，保留原有）=====================
print("===== 英特尔 Arrow Lake-P 核显加速配置 =====")
torch.set_num_threads(8)
torch.backends.mkldnn.enabled = True
torch.backends.mkldnn.benchmark = True
torch.backends.openmp.enabled = True
torch.backends.openmp.omp_num_threads = 8
torch.set_float32_matmul_precision('high')
device = torch.device("cpu")  # MKLDNN自动调度核显
# 加载CLIP模型+推理配置
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()  # 推理模式，提速30%
torch.set_grad_enabled(False)  # 全局关闭梯度，省显存
# 打印加速验证
print(f"✅ MKLDNN核显加速生效: {torch.backends.mkldnn.is_available()}")
print(f"✅ CLIP模型: ViT-B/32 | 线程数:8 | 匹配阈值:{CONFIDENCE_THRESHOLD*100}%")
print(f"✅ 全局DB表：{DB_TABLE_NAME} | 置信度过滤：{MIN_DETECTION_CONFIDENCE*100}%")
print(f"✅ 深度当前场景固定规则：1张当前图 + 7张历史最新图 = 8张，单批次直连大模型")
print("="*70 + "\n")

# -------------------------- CLIP异步推理配置（核显专属，修复队列阻塞） --------------------------
QUEUE_SIZE = 2
BATCH_SIZE = 32
img_queue = queue.Queue(maxsize=QUEUE_SIZE)
all_features = []
batch_lock = threading.Lock()

# 🔥 修改：统一使用detection_objects表，字段适配真实表结构
def filter_matched_image_paths(target_categories: List[str], db_path: str = DB_FILE_PATH) -> List[str]:
    """CLIP专用：DB类别筛选+时间窗口限制，去重返回图像路径
    统一使用detection_objects表，支持空列表（背景图）全量筛选
    """
    target_set: Set[str] = set([cat.strip().lower() for cat in target_categories])
    matched_image_map: dict = {}  # key:image_id 去重，value:original_path
    conn = None
    now = datetime.now()
    window_start = now - timedelta(hours=TIME_WINDOW_HOURS)
    # 日志：区分正常筛选/背景图全量筛选
    if target_set:
        print(f"📌 CLIP类别筛选：仅筛选 {TIME_WINDOW_HOURS} 小时内[{target_set}]相关图像", file=sys.stderr)
    else:
        print(f"📌 CLIP背景图模式：跳过类别筛选，取 {TIME_WINDOW_HOURS} 小时内所有去重图像", file=sys.stderr)

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        # 基础SQL：统一使用detection_objects表，查询必要字段
        base_sql = f"SELECT image_id, label_name, original_path, save_time FROM {DB_TABLE_NAME} WHERE 1=1"
        params = []
        # 时间窗口过滤（必加）
        base_sql += " AND save_time BETWEEN ? AND ?"
        params.extend([window_start.strftime('%Y-%m-%d %H:%M:%S'), now.strftime('%Y-%m-%d %H:%M:%S')])
        # 全局置信度过滤（必加）
        base_sql += " AND confidence >= ?"
        params.append(MIN_DETECTION_CONFIDENCE)
        # 多目标筛选：有类别则加，无则跳过（背景图）
        if target_set:
            # 中文转英文（适配DB中label为英文）
            en_targets = [COCO80_CN2EN.get(cat, cat) for cat in target_set if cat in COCO80_CN2EN]
            if en_targets:
                placeholders = ', '.join(['?'] * len(en_targets))
                base_sql += f" AND label_name IN ({placeholders})"
                params.extend(en_targets)
        # 排序
        base_sql += " ORDER BY image_id ASC"
        cursor.execute(base_sql, params)
        rows = cursor.fetchall()

        if not rows:
            print(f"⚠️  数据库{DB_TABLE_NAME}表无符合条件数据", file=sys.stderr)
            return []

        for row in rows:
            image_id = row["image_id"]
            original_path = row["original_path"].strip() if row["original_path"] else ""
            # 校验路径和image_id，去重
            if not original_path or image_id in matched_image_map:
                continue
            # 去重存储
            matched_image_map[image_id] = original_path

    except sqlite3.Error as e:
        print(f"❌ 数据库操作失败: {str(e)}", file=sys.stderr)
        return []
    finally:
        if conn:
            conn.close()

    matched_paths: List[str] = list(matched_image_map.values())
    # 日志：打印筛选结果
    if target_set:
        print(f"\n📊 DB类别筛选完成：目标类别={target_categories} | 匹配{len(matched_paths)}张唯一图像", file=sys.stderr)
    else:
        print(f"\n📊 DB背景图筛选完成：全量取图 | 匹配{len(matched_paths)}张唯一图像", file=sys.stderr)
    return matched_paths

# 🔥 修改：统一表名，字段适配detection_objects
def get_all_db_rows_by_path(image_path: str, db_path: str = DB_FILE_PATH) -> List[Dict[str, Any]]:
    """CLIP专用：根据图像路径查询detection_objects表所有对应行数据"""
    conn = None
    db_rows = []
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        sql = f"SELECT * FROM {DB_TABLE_NAME} WHERE original_path = ? ORDER BY save_time DESC"
        cursor.execute(sql, (image_path,))
        rows = cursor.fetchall()
        db_rows = [dict(row) for row in rows]
    except sqlite3.Error as e:
        print(f"❌ 路径查DB失败：{str(e)} | 路径：{image_path[-50:]}", file=sys.stderr)
    finally:
        if conn:
            conn.close()
    return db_rows

# ====================== 新增：JSON序列化兼容函数（核心修复float32报错）=====================
def convert_to_json_serializable(data: Any) -> Any:
    """递归转换非JSON可序列化类型为Python基础类型，处理numpy/bytes等"""
    if isinstance(data, (np.float32, np.float64, np.float16)):
        return float(data)
    elif isinstance(data, (np.int64, np.int32, np.uint32, np.uint64)):
        return int(data)
    elif isinstance(data, bytes):
        return data.decode('utf-8', errors='ignore')
    elif isinstance(data, dict):
        return {k: convert_to_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_json_serializable(item) for item in data]
    else:
        return data

# ====================== CLIP工具函数（异步+核显加速+异常兜底，修复队列和图片关闭）=====================
def load_and_preprocess(img_path: str) -> torch.Tensor:
    """CPU加载图片，异常返回全0张量，使用with上下文关闭图片"""
    try:
        with Image.open(img_path).convert("RGB") as img:
            return preprocess(img)
    except Exception as e:
        print(f"⚠️  图片处理失败: {img_path[-50:]} | {str(e)}", file=sys.stderr)
        return torch.zeros(3, 224, 224)

def infer_batch(batch_imgs: List[torch.Tensor]) -> None:
    """核显批量推理，提取特征"""
    global all_features
    batch_tensor = torch.stack(batch_imgs)
    with torch.no_grad():
        batch_features = model.encode_image(batch_tensor)
        batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
    batch_np = batch_features.detach().cpu().numpy()
    with batch_lock:
        all_features.append(batch_np)

def producer(img_paths: List[str]) -> None:
    """生产者：8线程加载图片，喂给核显，修复队列put阻塞"""
    with ThreadPoolExecutor(max_workers=8) as executor:
        for img_tensor in executor.map(load_and_preprocess, img_paths):
            img_queue.put(img_tensor, block=True, timeout=5.0)  # 增加阻塞和超时
    img_queue.put(None)  # 生产完成信号

def consumer() -> None:
    """消费者：核显凑批次推理，无空等"""
    batch_imgs = []
    while True:
        img_tensor = img_queue.get()
        if img_tensor is None:
            if batch_imgs:
                infer_batch(batch_imgs)
            break
        batch_imgs.append(img_tensor)
        if len(batch_imgs) >= BATCH_SIZE:
            infer_batch(batch_imgs)
            batch_imgs = []

def extract_image_features_batch(img_paths: List[str]) -> np.ndarray:
    """封装异步特征提取，返回numpy特征数组"""
    global all_features
    all_features = []
    prod_thread = threading.Thread(target=producer, args=(img_paths,))
    cons_thread = threading.Thread(target=consumer)
    prod_thread.start()
    cons_thread.start()
    prod_thread.join()
    cons_thread.join()
    return np.concatenate(all_features, axis=0) if all_features else np.array([])

def run_clip_matching(query_img_path: str, candidate_img_paths: List[str]) -> List[Dict[str, Any]]:
    """CLIP核显匹配，返回TOP8历史匹配大列表（含rank/confidence/db_rows），增加路径校验"""
    if not candidate_img_paths:
        print(f"\n❌ 无候选图像，终止CLIP匹配", file=sys.stderr)
        return []
    # 校验查询图是否存在
    if not os.path.exists(query_img_path):
        print(f"\n❌ 查询图不存在：{query_img_path[-50:]}，终止CLIP匹配", file=sys.stderr)
        return []

    print(f"\n" + "="*70, file=sys.stderr)
    print(f"===== 🚀 开始CLIP核显匹配（共{len(candidate_img_paths)}张候选图） =====", file=sys.stderr)
    print("="*70 + "\n", file=sys.stderr)
    total_start = time.time()

    try:
        with Image.open(query_img_path).convert("RGB") as query_img:  # with上下文关闭图片
            query_img_tensor = preprocess(query_img).unsqueeze(0)
        with torch.no_grad():
            query_feature = model.encode_image(query_img_tensor)
            query_feature = query_feature / query_feature.norm(dim=-1, keepdim=True)
        query_feature = query_feature.detach().cpu().numpy()[0]
        print(f"✅ 查询图特征提取完成", file=sys.stderr)
    except Exception as e:
        print(f"❌ 查询图特征提取失败：{str(e)}", file=sys.stderr)
        return []

    candidate_features = extract_image_features_batch(candidate_img_paths)
    if candidate_features.size == 0:
        print("❌ 候选图特征提取失败", file=sys.stderr)
        return []
    print(f"✅ 候选图特征提取完成 | 维度：{candidate_features.shape}", file=sys.stderr)

    similarities = np.dot(candidate_features, query_feature)
    match_pairs = list(zip(candidate_img_paths, similarities))
    match_pairs.sort(key=lambda x: x[1], reverse=True)
    qualified_pairs = [(path, sim) for path, sim in match_pairs if sim >= CONFIDENCE_THRESHOLD and os.path.exists(path)]
    top_qualified_pairs = qualified_pairs[:TOP_K]

    if not top_qualified_pairs:
        print(f"❌ 无CLIP置信度≥{CONFIDENCE_THRESHOLD*100}%的有效匹配结果", file=sys.stderr)
        return []

    history_match_list = []
    for idx, (img_path, confidence) in enumerate(top_qualified_pairs, 1):
        db_all_rows = get_all_db_rows_by_path(img_path)
        history_match_list.append({
            "rank": idx,
            "image_path": img_path,
            "confidence": round(confidence, 4),
            "db_rows": db_all_rows
        })
        short_path = img_path[-30:] if len(img_path) > 30 else img_path
        print(f"TOP{idx:2d} | 路径：{short_path:30s} | 置信度：{confidence:.4f} ({confidence*100:.2f}%)", file=sys.stderr)

    total_cost = round((time.time() - total_start) * 1000, 2)
    print(f"\n✅ CLIP匹配完成 | 总耗时：{total_cost} ms | 生成{len(history_match_list)}条历史匹配数据", file=sys.stderr)
    print("="*70, file=sys.stderr)
    return history_match_list

# ====================== 核心封装函数（当前数据+历史数据 整合：新增实时XYZ坐标）=====================
def package_final_data(target_categories: List[str], target_xyz: List[List[float]], history_match_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """最终封装：当前场景数据（类别+时间+实时XYZ坐标） + 历史匹配数据，兼容空类别/空XYZ（背景图）"""
    current_scene_data = {
        "target_categories": target_categories if target_categories else ["无有效目标（背景图）"],
        "target_xyz_coords": target_xyz if target_xyz else [],
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "scene_type": "背景图" if not target_categories else "正常场景"  # 新增：标记场景类型，方便大模型推理
    }
    final_package_data = {
        "current_scene": current_scene_data,
        "history_matches": history_match_list
    }
    final_package_data = convert_to_json_serializable(final_package_data)
    return final_package_data

# =============================================================================
# 主服务端节点：整合原有所有逻辑 + 深度当前场景【8张图单批次】核心重构
# =============================================================================
class IntentClassifyAnswerServer(Node):
    def __init__(self):
        super().__init__('intent_classify_answer_server')
        # 创建Action服务端
        self._action_server = ActionServer(
            self,
            ApiAction,
            'intent_classify',
            self.execute_callback
        )
        # 深度操作状态标识：控制反馈线程启停
        self.is_depth_retrieving = False
        self.total_start = 0.0  # 全局开始时间（供反馈线程）

        # 唯一ROS2客户端：简单/深度当前场景 共用 /get_capture_labels_paths 服务
        self.common_ros2_client = self.create_client(Trigger, "/get_capture_labels_paths")
        max_retry = 10
        retry_count = 0
        while not self.common_ros2_client.wait_for_service(timeout_sec=1.0) and retry_count < max_retry:
            self.get_logger().warn(f"等待共用ROS2服务端 /get_capture_labels_paths 上线...（{retry_count+1}/{max_retry}）")
            retry_count += 1
        if retry_count >= max_retry:
            self.get_logger().fatal("❌ 超过最大重试次数，未连接到共用ROS2服务端！")
            sys.exit(1)
        self.get_logger().info("✅ 已连接共用ROS2数据服务端（简单/深度当前场景共用）！")

        # 初始化通义千问API+校验DB/表
        self._init_dashscope_api()
        self._check_db_and_table()  # 简化表校验，仅校验目标表

        # 初始化日志提示（突出8张图固定规则）
        self.get_logger().info("="*60)
        self.get_logger().info("📌 意图分类 + 多模态推理服务端（CLIP+8张图单批次深度推理版 + 时间筛选）")
        self.get_logger().info(f"🔧 支持意图：{SUPPORTED_INTENTS}")
        self.get_logger().info(f"⚡ 共用服务：/get_capture_labels_paths（简单/深度当前场景共用）")
        self.get_logger().info(f"📂 DB路径：{DB_FILE_PATH} | 全局统一表：{DB_TABLE_NAME}")
        self.get_logger().info(f"🔍 CLIP配置：ViT-B/32 | 阈值{CONFIDENCE_THRESHOLD*100}% | TOP{TOP_K} | 核显加速")
        self.get_logger().info(f"📍 深度当前场景核心规则：1张当前实时图 + 7张历史最新图 = 8张，单批次直连大模型")
        self.get_logger().info(f"🔍 深度操作支持：{len(SUPPORTED_TARGETS)}类目标（含人/物）")
        self.get_logger().info(f"🖼️  兼容背景图：无目标/空XYZ时自动适配，深度场景正常推理")
        self.get_logger().info("="*60 + "\n")

    def _init_dashscope_api(self):
        """初始化通义千问API"""
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key or len(self.api_key) < 30:
            self.get_logger().fatal("❌ 环境变量 DASHSCOPE_API_KEY 未设置或格式错误！")
            self.get_logger().fatal("👉 export DASHSCOPE_API_KEY=你的阿里云API_KEY")
            sys.exit(1)
        dashscope.api_key = self.api_key
        self.get_logger().info("✅ 通义千问API Key 初始化成功\n")

    def _check_db_and_table(self):
        """校验DB文件+全局统一表detection_objects是否存在"""
        if not os.path.exists(DB_FILE_PATH):
            self.get_logger().fatal(f"❌ DB文件不存在: {DB_FILE_PATH}")
            sys.exit(1)
        # 校验表是否存在
        conn = None
        try:
            conn = sqlite3.connect(DB_FILE_PATH)
            cursor = conn.cursor()
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{DB_TABLE_NAME}';")
            if not cursor.fetchone():
                self.get_logger().fatal(f"❌ 全局统一表 {DB_TABLE_NAME} 不存在于DB中！")
                sys.exit(1)
            self.get_logger().info("✅ DB文件及全局统一表校验通过！")
        except sqlite3.Error as e:
            self.get_logger().fatal(f"❌ 校验DB/表失败: {str(e)}")
            sys.exit(1)
        finally:
            if conn:
                conn.close()

    # ---------------------- 共用：调用ROS2服务获取当前数据（兼容空类别/空XYZ）----------------------
    def get_common_current_data(self) -> Optional[Dict[str, Any]]:
        """
        简单/深度当前场景共用：调用/get_capture_labels_paths获取当前实时数据
        核心：空类别/空XYZ不再返回None，视为正常背景图数据
        """
        req = Trigger.Request()
        future = self.common_ros2_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if future.result() is None:
            self.get_logger().error("❌ 调用共用ROS2服务超时（5秒），无法获取当前数据")
            return None
        response = future.result()
        if not response.success:
            self.get_logger().error(f"❌ 共用ROS2服务执行失败：{response.message}")
            return None

        # 解析服务端返回的JSON
        try:
            result_data = json.loads(response.message)
            target_categories = result_data.get("labels", [])
            target_xyz_coords = result_data.get("xyz_coords", [])
            image_paths = result_data.get("image_paths", [])
            
            # 格式兜底与类型校验
            target_categories = target_categories if isinstance(target_categories, list) else []
            target_xyz_coords = target_xyz_coords if isinstance(target_xyz_coords, list) else []
            image_paths = [p.strip() for p in image_paths if isinstance(image_paths, list) and p.strip()]
            query_img_path = image_paths[0] if image_paths else ""

            # 仅当有类别但坐标不匹配时才警告，不终止
            if target_categories and len(target_categories) != len(target_xyz_coords):
                self.get_logger().warning(f"⚠️  共用服务返回数据异常：类别数({len(target_categories)})与坐标数({len(target_xyz_coords)})不匹配，按实际数据处理")

            # 解析坐标为扁平列表[x,y,z]
            parsed_xyz = []
            for xyz in target_xyz_coords:
                if isinstance(xyz, list) and len(xyz) >=1 and isinstance(xyz[0], list) and len(xyz[0])==3:
                    parsed_xyz.append(xyz[0])
                elif xyz:  # 非空但格式错误才警告
                    self.get_logger().warning(f"⚠️  共用服务坐标格式错误：{xyz}，跳过该坐标")

            # 仅当无图像路径时返回None，空类别/空坐标视为正常
            if not query_img_path or not os.path.exists(query_img_path):
                self.get_logger().error(f"❌ 共用服务返回无效查询图路径：{query_img_path}")
                return None

            # 日志区分正常场景/背景图
            if target_categories and parsed_xyz:
                self.get_logger().info(f"✅ 从共用服务获取当前数据：类别={target_categories}，坐标={parsed_xyz}，图像={query_img_path[-50:]}")
            else:
                self.get_logger().info(f"✅ 从共用服务获取当前数据：背景图（无目标/空坐标），图像={query_img_path[-50:]}")

            return {
                "query_img_path": query_img_path,
                "target_categories": target_categories,
                "target_xyz": parsed_xyz
            }
        except Exception as e:
            self.get_logger().error(f"❌ 解析共用服务数据失败：{str(e)}")
            return None

    # ---------------------- 简单当前场景核心处理函数（保留原有）----------------------
    def _handle_simple_current_scene(self, user_question: str) -> str:
        """处理简单当前场景问题：CLIP结构化数据 + 大模型融合推理"""
        self.get_logger().info("🔍 开始处理简单当前场景问题，执行「CLIP数据获取+大模型融合推理」流程...")
        clip_structured_data = self._get_clip_structured_data()
        if not clip_structured_data:
            return "❌ 无法回答你的问题：未获取到有效的场景图像（服务调用失败/无图像路径）"
        llm_final_answer = self._llm_infer_with_clip_data(user_question, clip_structured_data)
        return llm_final_answer

    # ---------------------- 获取CLIP结构化数据（兼容背景图，保留原有）----------------------
    def _get_clip_structured_data(self) -> Optional[Dict[str, Any]]:
        """仅获取CLIP核显匹配后的结构化数据（字典，含实时XYZ），兼容背景图"""
        service_data = self.get_common_current_data()
        if not service_data:
            self.get_logger().error("❌ 获取CLIP数据失败：无法调用ROS2服务获取实时场景图像")
            return None
        
        query_img_path = service_data["query_img_path"]
        target_categories = service_data["target_categories"]
        target_xyz = service_data["target_xyz"]
        
        candidate_img_paths = filter_matched_image_paths(target_categories)
        history_match_list = run_clip_matching(query_img_path, candidate_img_paths)
        final_structured_data = package_final_data(target_categories, target_xyz, history_match_list)
        self.get_logger().info("✅ 成功获取CLIP结构化数据（兼容背景图），即将交给大模型融合推理")
        return final_structured_data

    # ---------------------- 大模型融合推理（Prompt适配背景图，保留原有）----------------------
    def _llm_infer_with_clip_data(self, user_question: str, clip_data: Dict[str, Any]) -> str:
        """简单当前场景：大模型融合CLIP数据+XYZ坐标推理"""
        self.get_logger().info(f"💬 调用大模型融合CLIP数据推理：问题={user_question[:50]}...")
        infer_start = time.time()
        clip_data_json = json.dumps(clip_data, ensure_ascii=False, indent=2)

        prompt = f"""你是专业的智能场景分析助手，负责结合**当前数据和过去数据进行逻辑推理“
请严格遵守以下规则进行分析和回答：
1. 推理依据：仅基于下方提供的【CLIP结构化数据】，不臆造任何信息，数据不足时明确说明；
2. 关键数据标记解读：
   - scene_type: 背景图 → 当前画面无任何有效目标，只有背景；正常场景 → 当前画面检测到目标；
   - target_categories: ["无有效目标（背景图）"] → 确认当前是背景图，无目标；
   - current_scene.target_xyz_coords: 空列表 → 无目标的3D坐标；
   - history_matches: CLIP核显匹配的TOP8历史相似图像（rank=匹配排名，confidence=匹配置信度，代表历史是否见过该目标）；
   - 背景图就是当前帧本来只检测到目标物体，但是该物体被拿走了，该图中就没有别的被检测到的物体了
3. 严格遵守以下规则：
        1. 以用户为的问题为核心，只依据给定的CLIP数据进行推理，**严禁编造、脑补任何数据**。
        2. 若数据中没有用户询问的目标信息，直接说明“未检测到{{目标物品}}相关信息，无法判断”。
        3. 若当前场景为背景图、无有效目标：
        - 有历史匹配 → 说明目标不在当前视野，可引用历史匹配信息；
        - 无历史匹配 → 说明从未见过该目标，当前也无目标。
        4. 若为正常场景且检测到目标，结合实时XYZ坐标与历史匹配说明位置、是否移动等。
        5. 回答口语化、简洁，适合机器人播报，不输出代码、JSON、多余符号与无关解释。
        6. 若问题中有明显关联且在CLIP结构化数据中存在类别的目标，则一并进行记录分析  
        示例：我放在这的瓶子呢？是被谁拿走了吗，若有人拿走请描绘拿走瓶子的人 -> 瓶子不在当前视野里了，之前是在画面中央偏左的位置。有一个人在瓶子附近出现，这个人站在瓶子右侧约80厘米处，穿着普通衣物，具体特征无法识别
             我放在这的瓶子呢？有人拿走了吗？ ->  瓶子不在当前视野里了，但之前见过，当时附近有人，可能是被那个人拿走了 。此类问题不得捏造数据。
【用户的当前场景问题】：{user_question}

【CLIP结构化数据（含实时XYZ/背景图标记）】：
{clip_data_json}"""

        response = MultiModalConversation.call(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS_CURRENT_SCENE
        )

        infer_cost = round(time.time() - infer_start, 3)
        if response.status_code != 200:
            err_msg = f"大模型推理失败: {response.message[:60]}"
            self.get_logger().error(f"❌ {err_msg}，耗时{infer_cost}s")
            return f"处理失败：{err_msg}"
        
        llm_answer = response.output.choices[0].message.content[0]["text"].strip()
        if not llm_answer:
            llm_answer = "暂无相关场景数据，无法回答你的问题"
        
        self.get_logger().info(f"✅ 大模型融合CLIP数据推理完成，耗时{infer_cost}s | 回答：{llm_answer[:100]}...")
        return llm_answer

    # ---------------------- 1）意图分类（升级版：支持时间解析）----------------------
    def _llm_text_classify(self, question):
        # 获取当前精确时间，供大模型进行相对时间计算
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        prompt =f"""你是精准的意图与时间解析器。当前系统时间是：【{current_time_str}】。
请严格遵守以下要求分析用户问题：
1. 输出格式：严格且仅输出 "意图|目标|开始时间|结束时间" 的字符串格式。
   - 若用户未指定时间，时间字段留空。
   - 若用户指定时间（如“昨天下午3点到4点”），请基于当前系统时间计算出准确的 "YYYY-MM-DD HH:MM:SS" 格式。
   - 分隔符必须使用 | 
    2. 支持意图：{SUPPORTED_INTENTS}
    3. 深度操作支持目标：{SUPPORTED_TARGETS}
    4. 核心分类规则（优先级从高到低，严格执行）：
    - 视觉理解：询问**当前/现在/眼前/这里**看到了什么、画面内容、当前场景描述，属于实时画面理解。
        示例：描述一下现在看到的画面。→ 视觉理解
        示例：我现在在哪？→ 视觉理解
        示例：这里有什么？→ 视觉理解
    - 简单目标检索：询问**最近/刚才/之前**是否看到某目标，或历史记忆汇总，属于历史记忆查询。
        **格式：简单目标检索|目标名**（无具体目标则不加|）
        示例：看见杯子了吗？→ 简单目标检索|杯子
        示例：有没有人？→ 简单目标检索|人
    - 深度目标检索：寻找目标+带**具体静态特征**（颜色/形状/品牌）。
        示例：有没有看见红色的杯子？→ 深度目标检索|杯子
        示例：有没有看见百事可乐？-> 深度目标检索|瓶子
    - 简单当前场景问题：询问目标去向/变化（无特征）。
        示例：杯子去哪了？→ 简单当前场景问题|杯子
        示例：这个杯子有没有发生移动? ->简单当前场景问题|杯子
    - 深度当前场景问题：带特征的目标去向/变化。
        示例：红色的杯子去哪了？→ 深度当前场景问题|杯子
        示例：百事可乐去哪了？->深度当前场景问题|瓶子
    - 闲聊：无关查询。→ 闲聊
    5. 目标名提取规则：仅提取核心名词，禁止修饰词。
   
用户问题：{question}"""

        response = MultiModalConversation.call(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS_CLASSIFY
        )

        if response.status_code != 200:
            raise Exception(f"分类接口失败: {response.message[:60]}")

        res_text = response.output.choices[0].message.content[0]["text"].strip()
        # 清洗符号
        for char in ["【", "】", "(", ")", "[", "]", "{", "}"]:
            res_text = res_text.replace(char, "")
        res_text = res_text.strip()

        # 默认值
        intent_type = "闲聊"
        target_cn = None
        start_time = None
        end_time = None

        # 解析 "意图|目标|开始时间|结束时间"
        parts = res_text.split("|")
        if len(parts) >= 1: intent_type = parts[0].strip()
        if len(parts) >= 2: target_cn = parts[1].strip() if parts[1].strip() else None
        if len(parts) >= 3: start_time = parts[2].strip() if parts[2].strip() else None
        if len(parts) >= 4: end_time = parts[3].strip() if parts[3].strip() else None

        # 目标合法性校验
        if target_cn and target_cn not in SUPPORTED_TARGETS:
            self.get_logger().warning(f"⚠️  目标[{target_cn}]不在支持列表")
            return "闲聊", None, None, None

        if intent_type in SUPPORTED_INTENTS:
            return intent_type, target_cn, start_time, end_time
        else:
            return "闲聊", None, None, None

    # ---------------------- 2）深度操作专属：持续反馈线程（保留原有）----------------------
    def _depth_retrieval_feedback_thread(self, goal_handle, feedback):
        """深度检索/深度当前场景 持续向客户端发送feedback"""
        self.get_logger().info("🔍 启动深度操作持续反馈线程")
        while self.is_depth_retrieving and rclpy.ok() and goal_handle.is_active:
            try:
                feedback.feedback_msg = "正在深度处理中...（1张当前+7张历史图，单批次多模态推理）"
                feedback.elapsed_time = round(time.time() - self.total_start, 3)
                goal_handle.publish_feedback(feedback)
            except Exception as e:
                self.get_logger().warning(f"⚠️  反馈线程发布消息失败：{str(e)[:50]}")
            time.sleep(DEPTH_RETRIEVAL_FEEDBACK_INTERVAL)
        self.get_logger().info("🔍 深度操作持续反馈线程停止")

    # ---------------------- 3）闲聊专用 - 纯文本对话（保留原逻辑）----------------------
    def _llm_chat(self, question):
        prompt = f"""你是一个友好的智能机器人助手，负责日常闲聊对话，回答简洁、自然、易懂，适配口语化交流，简洁为首要目标。
用户的闲聊问题：{question}
要求：直接回答问题，无需多余前缀，回答长度适中"""
        
        self.get_logger().info(f"💬 调用大模型处理闲聊问题: {question[:50]}...")
        chat_start = time.time()
        response = MultiModalConversation.call(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS_CHAT
        )
        chat_cost = round(time.time() - chat_start, 3)

        if response.status_code != 200:
            raise Exception(f"闲聊接口调用失败: {response.message[:60]}")
        
        chat_ans = response.output.choices[0].message.content[0]["text"].strip()
        if not chat_ans:
            chat_ans = "我不太明白你的意思呢，换个问题问问吧～"
        
        self.get_logger().info(f"✅ 闲聊回答生成完成，耗时{chat_cost}s")
        return chat_ans
    
    # ---------------------- 简单目标检索查询函数（支持时间过滤）----------------------
    def _filter_clip_target_data(self, target_cn, start_time=None, end_time=None):
        """简单目标检索专属：查询DB，支持自定义时间段，默认72小时"""
        now = datetime.now()
        
        # 确定时间窗口
        if start_time and end_time:
            search_start = start_time
            search_end = end_time
            self.get_logger().info(f"🕒 启用精确时间筛选：{search_start} 至 {search_end}")
        else:
            search_start = (now - timedelta(hours=TIME_WINDOW_HOURS)).strftime('%Y-%m-%d %H:%M:%S')
            search_end = now.strftime('%Y-%m-%d %H:%M:%S')

        target_en = COCO80_CN2EN.get(target_cn, target_cn)
        matched_data = []
        conn = None

        try:
            conn = sqlite3.connect(DB_FILE_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            sql = f"""SELECT save_time, label_name, confidence, cam_x, cam_y, cam_z, original_path, image_id
                    FROM {DB_TABLE_NAME}
                    WHERE label_name = ? 
                    AND confidence >= ?
                    AND save_time BETWEEN ? AND ?
                    ORDER BY save_time DESC"""
            cursor.execute(sql, (
                target_en,
                MIN_DETECTION_CONFIDENCE,
                search_start,
                search_end
            ))
            rows = cursor.fetchall()

            if not rows:
                if start_time:
                    raise Exception(f"在 {start_time} 到 {end_time} 期间未找到【{target_cn}】数据")
                else:
                    raise Exception(f"表{DB_TABLE_NAME}中无【{target_cn}】72小时有效数据")

            # 按image_id去重
            image_id_set = set()
            for row in rows:
                image_id = row["image_id"]
                original_path = row["original_path"].strip() if row["original_path"] else ""
                if image_id in image_id_set or not original_path or not os.path.exists(original_path):
                    continue
                image_id_set.add(image_id)

                matched_data.append({
                    "time_str": row["save_time"].strip(),
                    "x": float(row["cam_x"]) if row["cam_x"] else 0.0,
                    "y": float(row["cam_y"]) if row["cam_y"] else 0.0,
                    "z": float(row["cam_z"]) if row["cam_z"] else 0.0,
                    "confidence": float(row["confidence"]) if row["confidence"] else 0.0,
                    "path": original_path
                })

            if not matched_data:
                raise Exception(f"该时段内【{target_cn}】无有效去重数据")

        except sqlite3.Error as e:
            raise Exception(f"表{DB_TABLE_NAME}操作失败: {str(e)[:50]}")
        finally:
            if conn:
                conn.close()

        self.get_logger().info(f"✅ 简单目标检索完成：目标【{target_cn}】| 有效数据{len(matched_data)}条")
        return matched_data

    # ---------------------- 简单目标检索核心处理函数（升级版）----------------------
    def _handle_simple_target_retrieval(self, user_question: str, target_cn: str, start_time=None, end_time=None) -> str:
        """简单目标检索核心处理：查询历史数据+大模型推理"""
        
        time_desc = f"在 {start_time} 到 {end_time} 期间" if start_time else "最近72小时内"
        self.get_logger().info(f"🔍 处理简单目标检索：目标【{target_cn}】，范围：{time_desc}...")
        
        try:
            # 传入时间参数
            clip_target_data = self._filter_clip_target_data(target_cn, start_time, end_time)
            if not clip_target_data:
                return f"❌ {time_desc}未检测到目标【{target_cn}】的相关数据"
            
            clip_data_json = json.dumps(clip_target_data, ensure_ascii=False, indent=2)
            
            prompt = f"""你是智能机器人记忆助手，负责回答用户「是否见过某目标」的简单检索问题。
请仅基于下方提供的【历史检测数据】进行推理：
数据时间范围：{time_desc}

规则：
1. 数据解读：time_str=检测时间，x/y/z=相对相机3D坐标，confidence=检测置信度，path=图像路径；
2. 回答要求：口语化、简洁明了，说明是否见过+最近一次出现的时间/大致位置；
3. 禁止输出JSON、代码，不臆造信息。

用户问题：{user_question}
目标：{target_cn}
历史检测数据：
{clip_data_json}"""
            
            response = MultiModalConversation.call(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS_INFER
            )
            
            if response.status_code != 200:
                raise Exception(f"大模型推理失败: {response.message[:60]}")
            
            llm_answer = response.output.choices[0].message.content[0]["text"].strip()
            self.get_logger().info(f"✅ 简单目标检索完成，回答：{llm_answer[:100]}...")
            return llm_answer if llm_answer else f"{time_desc}见过目标【{target_cn}】，具体信息可查看历史数据"
        
        except Exception as e:
            err_msg = str(e)[:80]
            self.get_logger().error(f"❌ 简单目标检索失败：{err_msg}")
            return f"❌ 检索目标【{target_cn}】失败：{err_msg}"

    # ---------------------- 4）通用目标筛选（深度检索用，支持时间过滤）----------------------
    def _filter_target_imgs_with_pose(self, target_cn, start_time=None, end_time=None):
        """深度检索专用：时空聚类去重，支持自定义时间段"""
        raw_items = [] 
        now = datetime.now()
        
        # 确定时间窗口
        if start_time and end_time:
            search_start = start_time
            search_end = end_time
            # 校验时间格式防止崩溃
            try:
                ts_start = parser.parse(search_start)
                ts_end = parser.parse(search_end)
            except:
                ts_start = now - timedelta(hours=TIME_WINDOW_HOURS)
                ts_end = now
        else:
            search_start = (now - timedelta(hours=TIME_WINDOW_HOURS)).strftime('%Y-%m-%d %H:%M:%S')
            search_end = now.strftime('%Y-%m-%d %H:%M:%S')
            ts_start = now - timedelta(hours=TIME_WINDOW_HOURS)
            ts_end = now

        target_en = COCO80_CN2EN[target_cn]
        conn = None
        try:
            conn = sqlite3.connect(DB_FILE_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            sql = f"""SELECT save_time, label_name, confidence, cam_x, cam_y, cam_z, original_path 
                    FROM {DB_TABLE_NAME} 
                    WHERE label_name = ? AND confidence >= ?
                    AND save_time BETWEEN ? AND ?
                    ORDER BY save_time DESC"""
            cursor.execute(sql, (
                target_en,
                MIN_DETECTION_CONFIDENCE,
                search_start,
                search_end
            ))
            rows = cursor.fetchall()

            if not rows:
                raise Exception(f"该时段内无【{target_cn}】有效数据")

            for row in rows:
                path = row["original_path"].strip() if row["original_path"] else ""
                if not path or not os.path.exists(path): continue
                
                time_str = row["save_time"].strip()
                try:
                    ts = parser.parse(time_str)
                    # 二次校验时间戳（DB已过滤，但parser需确保格式正确）
                    if ts < ts_start or ts > ts_end: continue
                except: continue

                conf = float(row["confidence"])
                
                raw_items.append({
                    "time_str": time_str,
                    "timestamp": ts.timestamp(),
                    "x": float(row["cam_x"]), 
                    "y": float(row["cam_y"]), 
                    "z": float(row["cam_z"]),
                    "confidence": conf,
                    "path": path
                })

        except sqlite3.Error as e:
            raise Exception(f"表{DB_TABLE_NAME}操作失败: {str(e)[:50]}")
        finally:
            if conn: conn.close()

        if not raw_items:
            raise Exception(f"该时段内无有效图片数据")

        # 时空聚类去重：0.5秒内的重复数据只取一张
        TIME_TOLERANCE = 0.5
        clustered_items = []
        current_group = [raw_items[0]]
        
        for i in range(1, len(raw_items)):
            prev_item = raw_items[i-1]
            curr_item = raw_items[i]
            time_diff = abs(prev_item["timestamp"] - curr_item["timestamp"])
            
            if time_diff <= TIME_TOLERANCE:
                current_group.append(curr_item)
            else:
                best_item = max(current_group, key=lambda x: x["confidence"])
                clustered_items.append(best_item)
                current_group = [curr_item]
        
        if current_group:
            best_item = max(current_group, key=lambda x: x["confidence"])
            clustered_items.append(best_item)

        self.get_logger().info(f"⚡ 数据压缩优化：原始{len(raw_items)}条 -> 采样后{len(clustered_items)}条")
        return clustered_items
    
    # ---------------------- 视觉理解业务处理函数（兼容背景图，保留原有）----------------------
    def _handle_visual_understanding(self, user_question: str) -> str:
        """视觉理解：结合当前实时图像Base64+结构化数据做多模态推理"""
        self.get_logger().info("🔍 处理视觉理解：获取当前实时场景数据+图像...")

        current_data = self.get_common_current_data()
        if not current_data:
            return "❌ 无法获取当前画面数据，请检查摄像头或数据服务是否正常运行。"

        query_img_path = current_data["query_img_path"]
        target_categories = current_data["target_categories"]
        target_xyz = current_data["target_xyz"]

        # 构造当前场景结构化信息
        current_scene_info = {
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "detected_objects": target_categories if target_categories else ["无任何有效目标"],
            "object_xyz_coords": target_xyz if target_xyz else ["无目标坐标（背景图）"],
        }
        current_info_json = json.dumps(current_scene_info, ensure_ascii=False, indent=2)

        # 当前图像转Base64
        img_base64 = ""
        try:
            img_base64 = self._img_to_base64(query_img_path)
            self.get_logger().info(f"✅ 当前图像转Base64完成：{query_img_path[-50:]}")
        except Exception as e:
            self.get_logger().warning(f"⚠️ 当前图像转Base64失败：{str(e)[:50]}，将仅基于结构化数据推理")

        # 构造多模态Prompt
        prompt = f"""你是机器人的视觉理解助手，负责真实描述当前画面看到的内容，回答用户关于“现在看到了什么”的问题。
    请严格遵守规则：
    1. 优先结合**图像视觉信息**描述画面，再补充**结构化数据**中的目标3D坐标；
    2. 描述要求：自然、口语化、简洁，先讲整体场景，再讲具体目标；
    3. 若图像转码失败/无图像信息，仅基于结构化数据描述；
    4. 若为背景图（无任何目标），直接说“当前画面只有背景，没有检测到任何有效目标”；
    5. 禁止编造信息，不输出JSON/代码/专业术语，适配机器人口语交流。

    【当前实时场景结构化数据】
    {current_info_json}

    【用户问题】：{user_question}"""

        # 构造多模态请求内容
        content = [{"text": prompt}]
        if img_base64:
            ext = os.path.splitext(query_img_path)[1].lower().replace('.', '')
            img_format = ext if ext in ["jpg", "jpeg", "png"] else "jpg"
            content.append({"image": f"data:image/{img_format};base64,{img_base64}"})

        self.get_logger().info("💬 调用大模型进行多模态视觉理解推理...")
        infer_start = time.time()

        # 调用通义千问多模态接口
        response = MultiModalConversation.call(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": content}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS_CURRENT_SCENE
        )

        infer_cost = round(time.time() - infer_start, 3)
        if response.status_code != 200:
            err_msg = f"视觉理解大模型调用失败: {response.message[:60]}"
            self.get_logger().error(f"❌ {err_msg}，耗时{infer_cost}s")
            return f"视觉理解失败：{err_msg}"

        llm_answer = response.output.choices[0].message.content[0]["text"].strip()
        if not llm_answer:
            llm_answer = "当前画面中未识别到明确目标，只有背景。"

        self.get_logger().info(f"✅ 视觉理解完成，耗时{infer_cost}s | 回答：{llm_answer[:100]}...")
        return llm_answer

    # ---------------------- 基础工具函数：图像转Base64（保留原有）----------------------
    def _img_to_base64(self, img_path):
        """图像转Base64，带路径校验和上下文关闭"""
        if not os.path.exists(img_path):
            raise Exception(f"图像文件不存在: {img_path[-50:]}")
        try:
            with open(img_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            raise Exception(f"图像读取失败 {img_path[-50:]}: {str(e)[:50]}")

    # ---------------------- 深度数据融合（兼容空类别/空XYZ，保留原有）----------------------
    def _fusion_depth_data(self, current_data: Dict[str, Any], history_depth_items: List[Dict[str, Any]], target_cn: str) -> Dict[str, Any]:
        """深度当前场景专用：融合当前实时数据 + 深度检索历史原始数据"""
        current_target_xyz = None
        if current_data["target_categories"]:
            for idx, cat in enumerate(current_data["target_categories"]):
                cat_clean = cat.strip().lower()
                if cat_clean == target_cn:
                    current_target_xyz = current_data["target_xyz"][idx] if idx < len(current_data["target_xyz"]) else None
                    break

        current_status = "有目标" if current_target_xyz else "无目标（背景图/未检测到）"
        self.get_logger().info(f"📊 当前目标状态：【{target_cn}】{current_status}")

        history_sorted = sorted(history_depth_items, key=lambda x: x["time_str"], reverse=True) if history_depth_items else []
        fusion_data = {
            "target_cn": target_cn,
            "current_data": {
                "query_img_path": current_data["query_img_path"],
                "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "current_xyz": current_target_xyz if current_target_xyz else [],
                "current_scene_categories": current_data["target_categories"] if current_data["target_categories"] else ["无有效目标（背景图）"],
                "current_target_status": current_status
            },
            "history_depth_data": history_sorted,
            "history_data_count": len(history_depth_items),
            "time_window_hours": TIME_WINDOW_HOURS,
            "min_detection_confidence": MIN_DETECTION_CONFIDENCE
        }

        self.get_logger().info(f"✅ 深度数据融合完成：目标【{target_cn}】| 当前状态{current_status} | 历史有效数据{len(history_depth_items)}条")
        return fusion_data

    # ---------------------- 🔥 核心重构：深度当前场景8张图单批次推理（删除所有分批/融合逻辑）----------------------
    def _handle_depth_current_scene(self, user_question: str, target_cn: str) -> str:
        """
        深度当前场景核心逻辑：1张当前图 + 7张历史最新图 = 8张，**单批次直接调用大模型**
        完全删除分批次、多线程、批次结果融合逻辑，流程极简：取图→转码→单请求→返回结果
        """
        self.get_logger().info(f"🔍 开始处理深度当前场景问题，目标【{target_cn}】| 固定规则：1张当前+7张历史=8张，单批次推理...")
        try:
            # 步骤1：获取当前实时数据（无图像直接返回失败）
            current_data = self.get_common_current_data()
            if not current_data or not current_data.get("query_img_path") or not os.path.exists(current_data["query_img_path"]):
                err_msg = "❌ 无法处理：调用ROS2服务获取当前实时图像失败（无有效路径）"
                self.get_logger().error(err_msg)
                return err_msg
            current_img_path = current_data["query_img_path"]
            self.get_logger().info(f"✅ 已获取当前实时图像：{current_img_path[-60:]}")

            # 步骤2：筛选目标历史深度数据（无数据则兼容，历史图补0）
            history_depth_items = []
            try:
                history_depth_items = self._filter_target_imgs_with_pose(target_cn)
            except Exception as e:
                self.get_logger().warning(f"⚠️ 无目标【{target_cn}】历史数据：{str(e)[:60]}，将仅使用当前1张图推理")

            # 步骤3：融合当前+历史结构化数据
            fusion_data = self._fusion_depth_data(current_data, history_depth_items, target_cn)
            fusion_data_json = json.dumps(fusion_data, ensure_ascii=False, indent=2)

            # 步骤4：固定取图 - 1张当前 + 7张历史最新（自动补全/截断）
            # 4.1 提取历史有效图像路径（去重、过滤无效）
            history_img_paths = []
            for item in history_depth_items:
                if item and "path" in item and item["path"] and os.path.exists(item["path"]):
                    history_img_paths.append(item["path"])
            # 4.2 历史图只取最新7张，不足则取全部，自动去重
            history_img_paths = history_img_paths[:FIXED_HISTORY_IMG]  # 截断为7张
            history_img_paths = list(dict.fromkeys(history_img_paths))  # 去重
            # 4.3 合并：1张当前 + 7张历史（总张数固定≤8）
            total_img_paths = [current_img_path] + history_img_paths
            self.get_logger().info(f"✅ 固定取图完成 | 当前图：1张 | 历史图：{len(history_img_paths)}张 | 总张数：{len(total_img_paths)}张")

            # 步骤5：构造单批次多模态请求（文本Prompt + 8张图Base64）
            request_content = self._build_single_batch_content(user_question, fusion_data_json, total_img_paths)
            if not request_content:
                err_msg = "❌ 构造多模态请求失败：无有效文本/图像数据"
                self.get_logger().error(err_msg)
                return err_msg

            # 步骤6：单批次调用大模型（核心：一次请求解决，无分批）
            self.get_logger().info(f"💬 单批次调用大模型推理（{len(total_img_paths)}张图）...")
            infer_start = time.time()
            final_answer = self._call_qwen_single_batch(request_content)
            infer_cost = round(time.time() - infer_start, 3)

            if final_answer == "推理失败":
                # 兜底：使用结构化数据生成回答
                final_answer = self._get_default_answer(fusion_data)
            self.get_logger().info(f"✅ 深度当前场景单批次推理完成 | 耗时{infer_cost}s | 回答：{final_answer[:100]}...")
            return final_answer

        except Exception as e:
            err_msg = str(e)[:150]
            self.get_logger().error(f"❌ 深度当前场景处理异常：{err_msg}")
            return f"❌ 深度场景处理失败：{err_msg}"

    # ---------------------- 辅助：构造单批次请求内容（文本+多张图）----------------------
    def _build_single_batch_content(self, user_question, fusion_data_json, img_paths):
        """构造单批次多模态请求内容：Prompt文本 + 所有图像Base64"""
        try:
            # 深度当前场景专用Prompt（引导大模型结合图像+结构化数据推理）
            prompt = f"""你是专业的机器人深度场景分析专家，负责回答带特征目标的**去向、是否被拿走、位置变化、是否移动**等核心问题。
以下是【全局结构化融合数据】和{len(img_paths)}张图像（第1张=当前实时图，后续=历史最新图），请结合图像视觉信息+结构化数据深度推理！
=== 核心推理依据 ===
{fusion_data_json}
=== 用户核心问题 ===
{user_question}
=== 严格回答规则 ===
1. 优先结合图像视觉特征（如目标颜色/形状/人物操作）+ 3D坐标分析；
2. 明确判定目标「是否被拿走/是否在当前视野/当前3D坐标/最后出现位置」；
3. 回答口语化、简洁明了，适配机器人终端，不输出JSON/代码/专业术语；
4. 无有效信息时直接说明，不臆造任何内容。"""
            
            # 构造请求：先文本，再逐张转Base64加图像
            content = [{"text": prompt}]
            for idx, img_path in enumerate(img_paths, 1):
                try:
                    b64 = self._img_to_base64(img_path)
                    ext = os.path.splitext(img_path)[1].lower().replace('.', '')
                    img_format = ext if ext in ["jpg", "jpeg", "png"] else "jpg"
                    content.append({"image": f"data:image/{img_format};base64,{b64}"})
                    self.get_logger().debug(f"✅ 第{idx}张图转Base64完成：{img_path[-30:]}")
                except Exception as e:
                    self.get_logger().warning(f"⚠️ 第{idx}张图转码失败：{img_path[-30:]} | {str(e)[:40]}")
                    continue

            return content if len(content) >= 1 else None
        except Exception as e:
            self.get_logger().error(f"❌ 构造请求内容失败：{str(e)[:60]}")
            return None

    # ---------------------- 辅助：单批次调用大模型（通义千问多模态）----------------------
    def _call_qwen_single_batch(self, content):
        """单批次调用通义千问多模态接口，返回推理结果"""
        try:
            response = MultiModalConversation.call(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": content}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS_DEPTH_CURRENT
            )
            if response.status_code != 200:
                self.get_logger().error(f"❌ 大模型调用失败：状态码{response.status_code} | {response.message[:60]}")
                return "推理失败"
            answer = response.output.choices[0].message.content[0]["text"].strip()
            return answer if answer else "推理失败"
        except Exception as e:
            self.get_logger().error(f"❌ 大模型调用异常：{str(e)[:80]}")
            return "推理失败"

    # ---------------------- 辅助：结构化数据兜底回答（大模型推理失败时用）----------------------
    def _get_default_answer(self, fusion_data):
        """基于结构化融合数据生成兜底回答，避免无结果"""
        target_cn = fusion_data["target_cn"]
        current_status = fusion_data["current_data"]["current_target_status"]
        history_count = fusion_data["history_data_count"]
        current_xyz = fusion_data["current_data"]["current_xyz"]
        history_data = fusion_data["history_depth_data"]

        if current_status == "有目标":
            return f"✅ 目标【{target_cn}】当前画面可检测到，相对相机3D坐标：X={current_xyz[0]:.2f}、Y={current_xyz[1]:.2f}、Z={current_xyz[2]:.2f}！"
        elif current_status == "无目标" and history_count > 0:
            last = history_data[0]
            last_time = last.get("time_str", "未知时间")
            last_xyz = f"X={last.get('x',0):.2f}、Y={last.get('y',0):.2f}、Z={last.get('z',0):.2f}"
            return f"❌ 目标【{target_cn}】当前未检测到，最后一次在{last_time}出现在{last_xyz}，判断已被拿走/移出视野！"
        else:
            return f"⚠️  目标【{target_cn}】当前未检测到，且近72小时无历史出现记录，无法判断状态！"

    # ---------------------- 深度目标检索原有逻辑（保留，未修改）----------------------
    def _infer_single_batch(self, user_question, batch_items, batch_idx, total_batch, target_cn):
        batch_start = time.time()
        content = []
        item_desc_list = []
        for i, item in enumerate(batch_items):
            desc = (f"图{i+1}：时间={item['time_str']}，相对相机位置 X={item['x']:.3f} Y={item['y']:.3f} Z={item['z']:.3f}，检测置信度={item['confidence']:.3f}")
            item_desc_list.append(desc)
        item_desc_str = "\n".join(item_desc_list)

        prompt = f"""用户问题：{user_question}（带特征的深度目标检索）
下面是一批检测图像，共{len(batch_items)}张，每张图的时间、相对相机3D位置和检测置信度信息如下（已按时间从早到晚排序）：
{item_desc_str}
请严格按照以下规则分析并输出，**必须完全遵守，不合并、不删减、不修改任何时间/位置坐标信息**：
1. 仅基于提供的图片和信息判断，重点匹配用户描述的目标特征；
2. 如果找到符合特征的目标【{target_cn}】，**每个目标的时间和位置都要单独成行、逐条输出**，固定格式：
   在{item['time_str']}，符合特征的目标【{target_cn}】出现在相对相机位置 X={item['x']:.3f} Y={item['y']:.3f} Z={item['z']:.3f}
3. 所有位置信息输出完成后，另起一行标注「【推理】」并基于特征、时间、位置推理分析问题并输出当前可能位置（信息不足则输出「当前位置无法判断」）；
4. 如果未找到符合特征的目标【{target_cn}】，仅输出：本批次未找到符合特征的目标，无其他任何内容。
输出要求：无额外文字、无序号、无标题，逻辑清晰。"""
        content.append({"text": prompt})

        for item in batch_items:
            b64 = self._img_to_base64(item["path"])
            ext = os.path.splitext(item["path"])[1].lower().replace('.', '')
            img_format = ext if ext in ["jpg", "jpeg", "png"] else "jpg"
            content.append({"image": f"data:image/{img_format};base64,{b64}"})

        response = MultiModalConversation.call(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": content}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS_INFER
        )

        if response.status_code != 200:
            raise Exception(f"第{batch_idx}/{total_batch}批次推理失败: {response.message[:60]}")

        ans = response.output.choices[0].message.content[0]["text"].strip()
        if not ans:
            raise Exception(f"第{batch_idx}/{total_batch}批次返回空结果")

        batch_cost = round(time.time() - batch_start, 3)
        self.get_logger().info(f"✅ 第{batch_idx}/{total_batch}批次处理完成，耗时{batch_cost}s")
        return ans

    def _fusion_batch_results(self, user_question, batch_answers, target_cn):
        fusion_prompt = f"""用户问题：{user_question}（带特征的深度目标检索）
下面是{len(batch_answers)}个图片批次的分析结果，所有批次已按时间从早到晚排序：
{chr(10).join(batch_answers)}
请严格按照以下规则汇总所有结果，生成最终答案，**必须完全遵守，不合并、不删减、不修改任何时间/位置坐标信息**：
1. 筛选有效信息：提取所有符合「在xxxx-xx-xx xx:xx:xx，符合特征的目标【{target_cn}】出现在相对相机位置 X=* Y=* Z=*」格式的行，删除无效行；
2. 整理位置信息：将所有有效行按时间从早到晚重新排序，每条信息单独占一行，无重复、无修改；
3. 生成最终推理：整理完所有位置信息后，标注「【最终推理】」并基于特征、位置变化、时间先后综合分析用户问题并输出当前可能位置，需有依据；
4. 无有效信息时，仅输出：未找到符合特征的目标【{target_cn}】相关检测图像信息；
5. 绝对禁止添加序号、标题、额外解释，仅按要求输出结果。"""

        fusion_response = MultiModalConversation.call(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": [{"text": fusion_prompt}]}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS_FUSION
        )

        if fusion_response.status_code != 200:
            raise Exception(f"结果融合失败: {fusion_response.message[:60]}")

        final_ans = fusion_response.output.choices[0].message.content[0]["text"].strip()
        if not final_ans:
            return f"未找到符合特征的目标【{target_cn}】相关检测图像信息"
        return final_ans

    def _llm_infer_images_with_pose(self, user_question, items, target_cn):
            batches = [items[i:i+FIXED_TOTAL_IMG] for i in range(0, len(items), FIXED_TOTAL_IMG)]
            total_batch = len(batches)
            self.get_logger().info(f"📤 共{len(items)}张有效图像，切分为{total_batch}批次进行深度检索推理")

            batch_answers = [None] * total_batch
            MAX_WORKERS = 6 
            
            from concurrent.futures import ThreadPoolExecutor, as_completed

            self.get_logger().info(f"⚡ 启动多线程加速，并发数: {MAX_WORKERS}...")
            parallel_start = time.time()

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_idx = {
                    executor.submit(
                        self._infer_single_batch, 
                        user_question, batch, idx, total_batch, target_cn
                    ): idx 
                    for idx, batch in enumerate(batches, 1)
                }
                
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        ans = future.result()
                        batch_answers[idx-1] = ans
                    except Exception as e:
                        err_msg = f"（第{idx}批次推理失败: {str(e)[:50]}）"
                        self.get_logger().error(f"❌ {err_msg}")
                        batch_answers[idx-1] = err_msg

            batch_answers = [ans for ans in batch_answers if ans]
            cost = time.time() - parallel_start
            self.get_logger().info(f"⚡ 并发推理完成，实际耗时: {cost:.2f}s")

            final_ans = self._fusion_batch_results(
                user_question=user_question,
                batch_answers=batch_answers,
                target_cn=target_cn
            )
            return final_ans

    # ---------------------- 无目标时的通用汇总（保留原有）----------------------
    def _handle_general_summary(self, question):
        """当用户问'看见什么了'且无具体目标时，汇总最近1小时的数据"""
        conn = sqlite3.connect(DB_FILE_PATH)
        cursor = conn.cursor()
        now = datetime.now()
        start_time = now - timedelta(hours=1)
        
        try:
            sql = f"""SELECT label_name FROM {DB_TABLE_NAME} 
                    WHERE confidence > {MIN_DETECTION_CONFIDENCE}
                    AND save_time BETWEEN ? AND ?
                    GROUP BY label_name"""
            cursor.execute(sql, (start_time.strftime('%Y-%m-%d %H:%M:%S'), now.strftime('%Y-%m-%d %H:%M:%S')))
            rows = cursor.fetchall()
            
            summary_list = []
            for row in rows:
                label_en = row[0]
                cn_name = COCO80_EN2CN.get(label_en, label_en)
                summary_list.append(cn_name)
            
            conn.close()

            if not summary_list:
                return "刚才这一个小时里，我没看见什么特别的东西。"
            
            prompt = f"用户问：'{question}'。最近1小时我看见了这些东西：{', '.join(summary_list)}。请用简短的口语汇总一下。"
            
            response = MultiModalConversation.call(
                model=MODEL_NAME, messages=[{"role": "user", "content": [{"text": prompt}]}]
            )
            return response.output.choices[0].message.content[0]["text"]
            
        except Exception as e:
            if conn: conn.close()
            return "检索记忆时出了点小差错，能再说一遍吗？"

    # ---------------------- 核心：按意图处理答案（核心：传入时间参数）----------------------
    def _intent_answer(self, intent, user_question, target_cn=None, start_time=None, end_time=None, goal_handle=None, feedback=None):
        """处理所有意图的业务逻辑"""
        if intent == "视觉理解":
            self.get_logger().info(f"🔍 触发视觉理解意图，问题: {user_question[:50]}...")
            return self._handle_visual_understanding(user_question)

        elif intent == "简单目标检索":
            self.get_logger().info(f"🔍 触发简单目标检索意图，目标: {target_cn}，问题: {user_question[:50]}...")
            return self._handle_simple_target_retrieval(user_question, target_cn, start_time, end_time)

        elif intent == "深度目标检索":
            self.get_logger().info(f"🔍 触发深度目标检索意图，目标: {target_cn}，问题: {user_question[:50]}...")
            self.is_depth_retrieving = True
            threading.Thread(
                target=self._depth_retrieval_feedback_thread,
                args=(goal_handle, feedback),
                daemon=True
            ).start()
            try:
                # 传入时间参数
                items = self._filter_target_imgs_with_pose(target_cn, start_time, end_time)
                ans = self._llm_infer_images_with_pose(user_question, items, target_cn)
                return ans
            finally:
                self.is_depth_retrieving = False

        elif intent == "简单当前场景问题":
            return self._handle_simple_current_scene(user_question)

        elif intent == "深度当前场景问题":
            self.get_logger().info(f"🔍 触发深度当前场景问题，目标: {target_cn}，问题: {user_question[:50]}...")
            self.is_depth_retrieving = True
            threading.Thread(
                target=self._depth_retrieval_feedback_thread,
                args=(goal_handle, feedback),
                daemon=True
            ).start()
            try:
                return self._handle_depth_current_scene(user_question, target_cn)
            finally:
                self.is_depth_retrieving = False

        elif intent == "闲聊":
            self.get_logger().info(f"💬 触发闲聊意图，用户问题: {user_question[:50]}...")
            return self._llm_chat(user_question)

        else:
            raise Exception(f"不支持的意图: {intent}，仅支持{SUPPORTED_INTENTS}")

    # ---------------------- Action 主回调（适配所有意图，升级参数接收）----------------------
    def execute_callback(self, goal_handle):
        self.total_start = time.time()
        user_q = goal_handle.request.user_question.strip()
        feedback = ApiAction.Feedback()
        result = ApiAction.Result()

        self.get_logger().info(f"📥 收到客户端请求: {user_q}")

        if not user_q:
            result.success = False
            result.intent = "闲聊"
            result.message = "请输入有效的问题（支持检索/当前场景/视觉理解/闲聊）"
            goal_handle.abort()
            return result

        try:
            # 阶段1：意图分类（接收4个返回值）
            feedback.feedback_msg = "正在进行意图与时间解析..."
            feedback.elapsed_time = round(time.time() - self.total_start, 3)
            goal_handle.publish_feedback(feedback)
            
            intent, target_cn, start_time, end_time = self._llm_text_classify(user_q)
            
            time_log = f" | 时间筛选：{start_time} 至 {end_time}" if start_time else ""
            self.get_logger().info(f"✅ 意图分类完成：{intent} | 目标：{target_cn}{time_log}")

            # 阶段2：按意图处理业务（传递所有参数）
            feedback.feedback_msg = f"正在{intent}处理中..."
            feedback.elapsed_time = round(time.time() - self.total_start, 3)
            goal_handle.publish_feedback(feedback)
            ans_start = time.time()
            
            final_ans = self._intent_answer(
                intent=intent,
                user_question=user_q,
                target_cn=target_cn,
                start_time=start_time, # 新增
                end_time=end_time,     # 新增
                goal_handle=goal_handle,
                feedback=feedback
            )
            
            ans_cost = round(time.time() - ans_start, 3)
            self.get_logger().info(f"✅ {intent}业务处理完成，耗时{ans_cost}s")

            # 阶段3：返回最终结果
            feedback.feedback_msg = f"{intent}处理完成，正在返回最终结果..."
            feedback.elapsed_time = round(time.time() - self.total_start, 3)
            goal_handle.publish_feedback(feedback)

            result.success = True
            result.intent = intent
            result.message = final_ans
            goal_handle.succeed()

            total_cost = round(time.time() - self.total_start, 3)
            self.get_logger().info(f"🎉 本次请求处理完成，总耗时: {total_cost}s\n")

        except Exception as e:
            self.is_depth_retrieving = False
            err = str(e)[:150]
            self.get_logger().error(f"❌ 处理过程中发生异常: {err}")
            feedback.feedback_msg = f"处理异常: {err[:80]}，任务终止"
            feedback.elapsed_time = round(time.time() - self.total_start, 3)
            goal_handle.publish_feedback(feedback)

            result.success = False
            result.intent = ""
            result.message = f"处理失败：{err}"
            goal_handle.abort()

        return result

# ---------------------- 主函数（精简，无冗余）----------------------
def main(args=None):
    rclpy.init(args=args)
    node = IntentClassifyAnswerServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.is_depth_retrieving = False
        node.get_logger().info("\n📤 收到终端中断信号，正在优雅停止服务端...")
    finally:
        node._action_server.destroy()
        node.destroy_node()
        rclpy.shutdown()
        print("\n✅ 视觉问答服务端已完全停止，资源释放完成！")

if __name__ == '__main__':
    main()