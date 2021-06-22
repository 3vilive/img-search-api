# coding: utf-8

import io
import dhash
import traceback
import faiss
import pickle
import time
import functools
import numpy as np
from os import path
from loguru import logger
from wand.image import Image
from flask import Flask, request, jsonify
from typing import List, Tuple, NamedTuple
from threading import Lock


# const
faiss_dimentions = 128
dhash_size = 8


app = Flask(__name__)


def singleton(klass):
    __ins = {}
    __lock = Lock()
    if app.config.get('gevent'):
        from gevent.lock import Semaphore
        __lock = Semaphore()

    @functools.wraps(klass)
    def wrapper(*args, **kwargs):
        if klass not in __ins:
            with __lock:
                if klass not in __ins:
                    print(f'ins {klass.__name__}')
                    __ins[klass] = klass(*args, **kwargs)

        return __ins[klass]

    return wrapper


class SearchResult(NamedTuple):
    dis: int
    idx: int

@singleton
class BinaryFlatIndex(object):
    def __init__(self) -> None:
        self.db_path = 'vec.dat'
        self.lock = Lock()

        if app.config.get('gevent_mode'):
            from gevent.lock import Semaphore
            self.lock = Semaphore()

        if path.exists(self.db_path):
            # load data from file
            logger.info(f'loading index db')
            self.vec_db = pickle.load(open(self.db_path, 'rb'))
        else:
            self.vec_db = np.array([], dtype='uint8')
            
        self.faiss_index = faiss.IndexBinaryFlat(faiss_dimentions)
        if len(self.vec_db) != 0:
            self.faiss_index.add(self.vec_db)

        self._auto_save_at = int(time.time())

    def allow_auto_save(self):
        if len(self.vec_db) == 0:
            return False

        now = int(time.time())
        if (now - self._auto_save_at) < 60:
            return False

        return True

    def save_to_db(self):
        logger.info(f'saving index db')
        pickle.dump(self.vec_db, open(self.db_path, 'wb'))


    def add_vec(self, vec: np.array):
        with self.lock:
            self.faiss_index.add(np.array([vec]))

            if len(self.vec_db) == 0:
                self.vec_db = vec
            else:
                self.vec_db = np.vstack((self.vec_db, vec))
            
            self.save_to_db()
            # time.sleep(5)

    def get_vec_by_index(self, idx: int) -> np.array:
        return self.vec_db[idx]
    
    def search_vec(self, vec: np.array, k: int=1) -> List[SearchResult]:
        dis, index = self.faiss_index.search(np.array([vec]), k=k)
        results = []

        for offset, _ in enumerate(index[0]):
            results.append(SearchResult(
                dis=dis[0][offset],
                idx=index[0][offset],
            ))

        return results


class utils(object):
    @classmethod
    def calc_dhash(cls, img: Image, size=8) -> Tuple[int, int]:
        return dhash.dhash_row_col(img, size=size)

    @classmethod
    def calc_dhash_bytes(cls, img: Image, size=8) -> bytes:
        row, col = cls.calc_dhash(img, size)
        return dhash.format_bytes(row, col, size)

    @staticmethod
    def format_hex(i: int):
        return f'{i:x}'

    @classmethod
    def bytes2vec(cls, bs: bytes) -> np.array:
        return np.array([i for i in bs], dtype='uint8')

    @classmethod
    def vec2hex(cls, vec: np.array) -> str:
        vec_bytes = vec.tobytes()
        i = int.from_bytes(vec_bytes, byteorder='big')
        return f'{i:x}'

    @classmethod
    def hex2vec(cls, hex: str) -> np.array:
        i = int(hex, base=16)
        bits_per_hash = dhash_size * dhash_size
        bs = i.to_bytes(bits_per_hash // 4, byteorder='big')
        return cls.bytes2vec(bs)

    @classmethod
    def img2vec(cls, img: Image, size=8):
        return cls.bytes2vec(
            cls.calc_dhash_bytes(img, size)
        )


class flask_utils(object):
    @classmethod
    def load_img(cls, field='image') -> Image:
        # save into img buff
        f = request.files[field]
        img_buff = io.BytesIO()
        f.save(img_buff)

        # load image
        img_buff.seek(0)
        img = Image(file=img_buff)
        img_buff.close()

        return img





@app.route('/ping')
def on_ping():
    return jsonify({
        'msg': 'pong',
    })


@app.route('/img/calc_img_dhash', methods=['POST'])
def on_calc_img_dhash():
    try:
        # load img
        img = flask_utils.load_img()

        # clac hash
        row_hash, col_hash = utils.calc_dhash(img, size=dhash_size)

        # resp
        return jsonify({
            'hash_hex': dhash.format_hex(row_hash, col_hash, dhash_size),
        })

    except Exception as e:
        logger.error(f'error: {e}\n{traceback.format_exc()}')
        return jsonify({
            'msg': 'error',
            'error': str(e),
            'tb':  traceback.format_exc(),
        })


@app.route('/faiss/add_img', methods=['POST'])
def on_faiss_add():
    try:
        # load img
        img = flask_utils.load_img()
        img_vec = utils.img2vec(img)

        idx = BinaryFlatIndex()
        idx.add_vec(img_vec)

        return jsonify({
            'index_size': idx.faiss_index.ntotal,
            'hash_hex': utils.vec2hex(img_vec),
        })

    except Exception as e:
        logger.error(f'error: {e}\n{traceback.format_exc()}')
        return jsonify({
            'msg': 'error',
            'error': str(e),
            'tb':  traceback.format_exc(),
        })


@app.route('/faiss/add_img_by_hex', methods=['POST'])
def on_faiss_add_by_hex():
    try:
        # get img hex
        hex = request.json.get('img_hex')
        img_vec = utils.hex2vec(hex)

        idx = BinaryFlatIndex()
        idx.add_vec(img_vec)

        return jsonify({
            'index_size': idx.faiss_index.ntotal,
            'hash_hex': utils.vec2hex(img_vec),
        })

    except Exception as e:
        logger.error(f'error: {e}\n{traceback.format_exc()}')
        return jsonify({
            'msg': 'error',
            'error': str(e),
            'tb':  traceback.format_exc(),
        })


@app.route('/faiss/search_img', methods=['POST'])
def on_faiss_search():
    try:
        # load img
        img = flask_utils.load_img()
        img_vec = utils.img2vec(img)

        # search
        idx = BinaryFlatIndex()
        results = idx.search_vec(img_vec, k=5)

        # build result
        output = []
        for result in results:
            hash_hex = utils.vec2hex(idx.get_vec_by_index(result.idx)) if result.idx != -1 else None
            output.append({
                'dis': int(result.dis),
                'index': int(result.idx),
                'hash_hex': hash_hex,
            })

        return jsonify({
            'data': output,
        })

    except Exception as e:
        logger.error(f'error: {e}\n{traceback.format_exc()}')
        return jsonify({
            'msg': 'error',
            'error': str(e),
            'tb':  traceback.format_exc(),
        })


@app.route('/faiss/search_img_by_hex', methods=['POST'])
def on_faiss_search_by_hex():
    try:
        # get img hex
        hex = request.json.get('img_hex')
        query_k = request.json.get('query_k', 1)
        img_vec = utils.hex2vec(hex)

        # search
        idx = BinaryFlatIndex()
        results = idx.search_vec(img_vec, query_k)

        # build result
        output = []
        for result in results:
            hash_hex = utils.vec2hex(idx.get_vec_by_index(result.idx)) if result.idx != -1 else None
            output.append({
                'dis': int(result.dis),
                'index': int(result.idx),
                'hash_hex': hash_hex,
            })

        return jsonify({
            'data': output,
        })

    except Exception as e:
        logger.error(f'error: {e}\n{traceback.format_exc()}')
        return jsonify({
            'msg': 'error',
            'error': str(e),
            'tb':  traceback.format_exc(),
        })


@app.route('/faiss/stat', methods=['GET'])
def on_faiss_stat():
    try:
        idx = BinaryFlatIndex()
        
        return jsonify({
            'index_size': idx.faiss_index.ntotal,
            'vec_db_size': idx.vec_db.shape,
        })

    except Exception as e:
        logger.error(f'error: {e}\n{traceback.format_exc()}')
        return jsonify({
            'msg': 'error',
            'error': str(e),
            'tb':  traceback.format_exc(),
        })
