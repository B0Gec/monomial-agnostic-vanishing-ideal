# https://qiita.com/suzuki-kei/items/b866fa4cd8f27d6b1ad5

# オブジェクトがハッシュ可能であるか判定する.
def hashable(x):
    try:
        hash(x)
        return True
    except TypeError:
        return False

# 関数を Memoize するデコレータ.
def memoize(callable):
    cache = {}
    def wrapper(*args, **kwargs):
        key = args + tuple(kwargs.items())
        if not hashable(key):
            return callable(*args, **kwargs)
        if key not in cache:
            cache[key] = callable(*args, **kwargs)
        return cache[key]
    return wrapper