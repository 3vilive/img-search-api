- [image-search-api](#image-search-api)
- [开发](#开发)
- [部署](#部署)
- [APIs](#apis)
  - [计算图片 dhash](#计算图片-dhash)
  - [faiss 状态](#faiss-状态)
  - [添加图片](#添加图片)
  - [根据 hex 添加图片](#根据-hex-添加图片)
  - [搜索图片](#搜索图片)
  - [根据 hex 搜索图片](#根据-hex-搜索图片)

# image-search-api

基于 dhash、faiss 的图片搜索 API


# 开发

```
FLASK_ENV=development flask run
```

# 部署

安装 minicoda:

```
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh
sh Miniconda3-py39_4.9.2-Linux-x86_64.sh
```

创建环境:

```
conda create -n img-dhash-faiss-api python=3.7
conda activate img-dhash-faiss-api
```

安装 ImageMagick

Ubuntu:

```
sudo apt-get install libmagickwand-dev -y
```

CentOS:

```
yum install ImageMagick-devel -y
```

安装 Python 相关依赖包：

```
pip install -r requirements.txt
```

安装 faiss:

```
conda install -c pytorch faiss-cpu
```


# APIs

## 计算图片 dhash 

- URL: `/img/calc_img_dhash`
- Method: `POST`
- Request Multipart Form:

```json
{
    "image": "Image Files Body",
}
```

- Response:

```json
{
  "hash_hex": "c6cec4d434a4644c07ef0a03fd350060"
}
```

## faiss 状态

- URL: `/faiss/stat`
- Method: `GET`
- Response:

```json
{
  "index_size": 2
}
```

## 添加图片

- URL: `/faiss/add_img`
- Method: `POST`
- Request Multipart Form:

```json
{
    "image": "Image Files Body",
}
```

- Response:

```json
{
  "hash_hex": "88e898b2d4ddd0cd7f4100e666b902d9",
  "index_size": 2
}
```


## 根据 hex 添加图片

- URL: `/faiss/add_img_by_hex`
- Method: `POST`
- Request Body:

```json
{
    "img_hex": "88e898b2d4ddd0cd7f4100e666b902d9",
}
```

- Response:

```json
{
  "hash_hex": "88e898b2d4ddd0cd7f4100e666b902d9",
  "index_size": 2
}
```

## 搜索图片

- URL: `/faiss/search_img`
- Method: `POST`
- Request Multipart Form:

```json
{
    "image": "Image Files Body",
}
```

- Response:

```json
{
  "data": [
    {
      "dis": 0,
      "hash_hex": "88e898b2d4ddd0cd7f4100e666b902d9",
      "index": 1
    }
  ]
}
```

## 根据 hex 搜索图片

- URL: `/faiss/search_img_by_hex`
- Method: `POST`
- Request Body:

```json
{
    "img_hex": "88e898b2d4ddd0cd7f4100e666b902d9",
}
```

- Response:

```json
{
  "data": [
    {
      "dis": 0,
      "hash_hex": "88e898b2d4ddd0cd7f4100e666b902d9",
      "index": 1
    }
  ]
}
```