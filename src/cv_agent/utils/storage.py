import base64
import json
import logging
import mimetypes
import os
import tempfile
import uuid
from io import BytesIO
import asyncio
import backoff
import httpx
import requests
from minio import Minio
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


# --- get_minio_config() and MinioManager remain unchanged ---
def get_minio_config():
    """Returns the hardcoded Minio configuration."""
    return {
        "ENDPOINT": os.environ.get("MINIO_ENDPOINT", "localhost:9000"),
        "ACCESS_KEY": os.environ.get("MINIO_ACCESS_KEY", ""),
        "SECRET_KEY": os.environ.get("MINIO_SECRET_KEY", ""),
        "BUCKET_NAME": os.environ.get("MINIO_BUCKET", "deep-research"),
        "SECURE": os.environ.get("MINIO_SECURE", "false").lower() == "true",
    }


def fetch_and_encode_url(image_url: str) -> str:
    """
    Fetches an image from a URL and encodes it as a base64 data URI.
    """
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Error on bad status (404, 500, etc.)

        image_data = response.content
        mime_type = response.headers.get("Content-Type")

        # Fallback if Content-Type is missing
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(image_url)

        if not mime_type:
            mime_type = "application/octet-stream"  # Default

        base64_string = base64.b64encode(image_data).decode("utf-8")
        data_uri = f"data:{mime_type};base64,{base64_string}"

        return data_uri

    except requests.RequestException as e:
        print(f"Error fetching or encoding URL {image_url}: {e}")
        # Return a simple text error if we can't get the image
        return f"Error: Failed to fetch image from {image_url}"


class MinioManager:
    def __init__(self, endpoint, access_key, secret_key, secure=False):
        self._client = Minio(endpoint, access_key, secret_key, secure=secure)

    def set_bucket_to_public_read(self, bucket_name):
        """Sets the S3 bucket policy to allow public read."""
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": "*",
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{bucket_name}/*"],
                }
            ],
        }
        try:
            self._client.set_bucket_policy(bucket_name, json.dumps(policy))
            logger.info("Set bucket '%s' policy to public read.", bucket_name)
        except Exception as e:
            logger.warning("Could not set bucket policy (may already be set): %s", e)

    # --- MODIFIED: Added retry logic ---
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def upload_file(self, bucket_name, object_name, file_path):
        """Uploads a file with retries."""
        if not self._client.bucket_exists(bucket_name):
            try:
                self._client.make_bucket(bucket_name)
                self.set_bucket_to_public_read(bucket_name)
            except Exception as e:
                # This can fail harmlessly if multiple processes try at once
                logger.warning("Bucket creation/policy set failed (may exist): %s", e)

        self._client.fput_object(bucket_name, object_name, file_path)

async def upload_pil_image_to_minio(image: Image.Image, sample_id: str) -> tuple[str, int, int]:
    """
    异步上传PIL图像到Minio，避免阻塞事件循环。
    Returns: (public_url, image_width, image_height)
    """
    temp_file_path = None
    try:
        # Get dimensions before conversion
        width, height = image.size

        # 图像处理部分保留在主线程（计算密集型，耗时较短通常可接受，也可以放线程池）
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        # 保存临时文件
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            image.save(tmp_file.name, format="JPEG", quality=85, optimize=True)
            temp_file_path = tmp_file.name

        # 【核心修改】定义一个同步的内部函数，包含所有 Minio 阻塞操作
        def _blocking_upload_task():
            config = get_minio_config()
            minio_client = MinioManager(
                endpoint=config.get("ENDPOINT"),
                access_key=config.get("ACCESS_KEY"),
                secret_key=config.get("SECRET_KEY"),
                secure=config.get("SECURE", False),
            )
            bucket_name = config.get("BUCKET_NAME")
            object_name = f"cvagent/initial/{sample_id}.jpg"

            # 这里的 upload_file 是同步阻塞的，现在它将在线程中运行
            minio_client.upload_file(
                bucket_name=bucket_name, object_name=object_name, file_path=temp_file_path
            )

            protocol = "https" if config.get("SECURE", False) else "http"
            endpoint = config.get("ENDPOINT")
            url = f"{protocol}://{endpoint}/{bucket_name}/{object_name}"

            logger.info("Uploaded initial image for sample %s to: %s", sample_id, url)
            return url

        # 【核心修改】将同步任务扔到线程池执行，并 await 结果
        loop = asyncio.get_running_loop()
        url = await loop.run_in_executor(executor, _blocking_upload_task)

        return url, width, height

    except Exception as e:
        logger.error("上传PIL图像失败 for sample %s: %s", sample_id, str(e))
        raise
    finally:
        # 清理临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass


@backoff.on_exception(backoff.expo, httpx.RequestError, max_tries=2)
async def download_image_to_pil(image_url: str) -> Image.Image:
    """Downloads an image from a URL and returns a PIL Image object, with retries."""
    try:
        async with httpx.AsyncClient() as client:
            # Set a reasonable timeout
            response = await client.get(image_url, timeout=30.0)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            return image
    except Exception as e:
        logger.error("Failed to download image from %s: %s", image_url, e)
        raise

executor = ThreadPoolExecutor(max_workers=5)


async def upload_new_artifact(image: Image.Image) -> tuple[str, int, int]:
    """
    异步非阻塞版本的 artifact 上传
    """
    temp_file_path = None  # 1. 预先定义，防止 finally 块报错
    width, height = 0, 0  # 预定义，防止异常时未绑定

    try:
        # Get dimensions before conversion
        width, height = image.size

        # --- 图像预处理逻辑 (补全) ---
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        # --- 创建临时文件 (补全) ---
        # delete=False 是必须的，因为我们要把文件名传给另一个线程去读取
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            image.save(tmp_file.name, format="JPEG", quality=70, optimize=True)
            temp_file_path = tmp_file.name  # 2. 在这里正确赋值

        # 定义一个同步的上传任务函数（闭包）
        def _sync_upload_task():
            # 这里是同步代码，将在线程池中运行，不会阻塞主循环
            config = get_minio_config()
            minio_client = MinioManager(
                endpoint=config.get("ENDPOINT"),
                access_key=config.get("ACCESS_KEY"),
                secret_key=config.get("SECRET_KEY"),
                secure=config.get("SECURE", False),
            )
            bucket_name = config.get("BUCKET_NAME")

            # 使用 UUID 生成唯一文件名
            object_name = f"cvagent/artifacts/{uuid.uuid4()}.jpg"

            # 3. 这里现在可以安全地访问 temp_file_path 了
            minio_client.upload_file(
                bucket_name=bucket_name, object_name=object_name, file_path=temp_file_path
            )

            protocol = "https" if config.get("SECURE", False) else "http"
            url = f"{protocol}://{config.get('ENDPOINT')}/{bucket_name}/{object_name}"
            return url

        # 关键修改：使用 run_in_executor 将阻塞操作扔到线程池
        loop = asyncio.get_running_loop()
        url = await loop.run_in_executor(executor, _sync_upload_task)

        logger.info("New artifact uploaded to Minio: %s", url)
        return url, width, height  # Return dimensions

    except Exception as e:
        logger.error("Uploading new artifact failed: %s", str(e))
        return "", 0, 0

    finally:
        # 4. 安全清理：检查变量是否已定义且文件是否存在
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass

def pil_to_base64_uri(image: Image.Image) -> str:
    """Converts a PIL Image object into a Base64 Data URI string."""

    # 确保图像是 RGB 格式，以兼容大多数模型和 Base64 编码
    if image.mode == "RGBA":
        # 如果是 RGBA，将其转换为 RGB (白色背景)
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    buffer = BytesIO()
    # 使用 JPEG 格式以减小 Base64 字符串的长度
    image.save(buffer, format="JPEG", quality=85, optimize=True)
    base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # 构造 Data URI 格式
    data_uri = f"data:image/jpeg;base64,{base64_string}"
    return data_uri
