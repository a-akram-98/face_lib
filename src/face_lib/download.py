"""
This file is imported with some modification from gdown repo, many thanks and support to them.
gdown repo: https://github.com/wkentaro/gdown
"""
from __future__ import print_function

import glob
import json
import os
import os.path as osp
import re
import shutil
import sys
import tempfile
import textwrap
import time

import requests
import six
import tqdm
import logging

from .parse_url import parse_url

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logging_handle = "[Face Library]"
logger = logging.getLogger(logging_handle)

CHUNK_SIZE = 512 * 1024  # 512KB
home = osp.expanduser("~")


if hasattr(textwrap, "indent"):
    indent_func = textwrap.indent
else:

    def indent_func(text, prefix):
        def prefixed_lines():
            for line in text.splitlines(True):
                yield (prefix + line if line.strip() else line)

        return "".join(prefixed_lines())


def get_url_from_gdrive_confirmation(contents):
    url = ""
    for line in contents.splitlines():
        m = re.search(r'href="(\/uc\?export=download[^"]+)', line)
        if m:
            url = "https://docs.google.com" + m.groups()[0]
            url = url.replace("&amp;", "&")
            break
        m = re.search("confirm=([^;&]+)", line)
        if m:
            confirm = m.groups()[0]
            url = re.sub(
                r"confirm=([^;&]+)", r"confirm={}".format(confirm), url
            )
            break
        m = re.search('"downloadUrl":"([^"]+)', line)
        if m:
            url = m.groups()[0]
            url = url.replace("\\u003d", "=")
            url = url.replace("\\u0026", "&")
            break
        m = re.search('<p class="uc-error-subcaption">(.*)</p>', line)
        if m:
            error = m.groups()[0]
            raise RuntimeError(error)
    if not url:
        raise RuntimeError(
            "Cannot retrieve the public link of the file. "
            "You may need to change the permission to "
            "'Anyone with the link', or have had many accesses."
        )
    return url


def download(
    url=None,
    output=None,
    quiet=False,
    proxy=None,
    speed=None,
    use_cookies=True,
    verify=True,
    id=None,
    fuzzy=False,
    resume=False,
):
    """Download file from URL.

    Parameters
    ----------
    url: str
        URL. Google Drive URL is also supported.
    output: str
        Output filename. Default is basename of URL.
    quiet: bool
        Suppress terminal output. Default is False.
    proxy: str
        Proxy.
    speed: float
        Download byte size per second (e.g., 256KB/s = 256 * 1024).
    use_cookies: bool
        Flag to use cookies. Default is True.
    verify: bool or string
        Either a bool, in which case it controls whether the server’s TLS
        certificate is verified, or a string, in which case it must be a path
        to a CA bundle to use. Default is True.
    id: str
        Google Drive's file ID.
    fuzzy: bool
        Fuzzy extraction of Google Drive's file Id. Default is False.
    resume: bool
        Resume the download from existing tmp file if possible.
        Default is False.

    Returns
    -------
    output: str
        Output filename.
    """
    if not (id is None) ^ (url is None):
        raise ValueError("Either url or id has to be specified")
    if id is not None:
        url = "https://drive.google.com/uc?id={id}".format(id=id)

    url_origin = url
    sess = requests.session()
    res = None
    # Load cookies
    cache_dir = osp.join(home, ".cache", "gdown")
    if not osp.exists(cache_dir):
        os.makedirs(cache_dir)
    cookies_file = osp.join(cache_dir, "cookies.json")
    if osp.exists(cookies_file) and use_cookies:
        with open(cookies_file) as f:
            cookies = json.load(f)
        for k, v in cookies:
            sess.cookies[k] = v

    if proxy is not None:
        sess.proxies = {"http": proxy, "https": proxy}
        logger.info("Using proxy:" + proxy)

    gdrive_file_id, is_gdrive_download_link = parse_url(url, warning=not fuzzy)

    if fuzzy and gdrive_file_id:
        # overwrite the url with fuzzy match of a file id
        url = "https://drive.google.com/uc?id={id}".format(id=gdrive_file_id)
        url_origin = url
        is_gdrive_download_link = True

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"  # NOQA
    }

    while True:
        try:
            if id is None:
                res = sess.get(url, headers=headers, stream=True, verify=verify, timeout= 100)
            else:
                res = sess.get(url, headers=headers, stream=True, verify=verify, timeout = 300)

        except requests.exceptions.ProxyError as e:
            logger.error("An error has occurred using proxy")
            return "Error"
        except requests.exceptions.ConnectionError as e:
            logger.error("An error has occurred connectiong to download server")
            return "Error"

        

        # Save cookies
        with open(cookies_file, "w") as f:
            cookies = [
                (k, v)
                for k, v in sess.cookies.items()
                if not k.startswith("download_warning_")
            ]
            json.dump(cookies, f, indent=2)

        if "Content-Disposition" in res.headers:
            # This is the file
            break
        if not (gdrive_file_id and is_gdrive_download_link):
            break

        # Need to redirect with confirmation
        try:
            url = get_url_from_gdrive_confirmation(res.text)
        except RuntimeError as e:
            logger.error("Access denied with the following error:")
            error = "\n".join(textwrap.wrap(str(e)))
            error = indent_func(error, "\t")
            logger.error("\n"+ error+"\n")
            logger.info("You may still be able to access the file from the browser")
            logger.info("\n\t"+url_origin+"\n")
            return "Error"

    if gdrive_file_id and is_gdrive_download_link:
        m = re.search('filename="(.*)"', res.headers["Content-Disposition"])
        filename_from_url = m.groups()[0]
    else:
        filename_from_url = osp.basename(url)

    if output is None:
        output = filename_from_url

    output_is_path = isinstance(output, six.string_types)
    if output_is_path and output.endswith(osp.sep):
        if not osp.exists(output):
            os.makedirs(output)
        output = osp.join(output, filename_from_url)

    if output_is_path:
        existing_tmp_files = glob.glob("{}*".format(output))
        if resume and existing_tmp_files:
            tmp_file = existing_tmp_files[0]
        else:
            resume = False
            tmp_file = tempfile.mktemp(
                suffix=tempfile.template,
                prefix=osp.basename(output),
                dir=osp.dirname(output),
            )
            tmp_file = osp.join(osp.dirname(output),osp.basename(output))
        f = open(tmp_file, "ab")
    else:
        tmp_file = None
        f = output

    if tmp_file is not None and f.tell() != 0:
        headers["Range"] = "bytes={}-".format(f.tell())
        if id is None:
            res = sess.get(url, headers=headers, stream=True, verify=verify, timeout = 100)
        else:
            res = sess.get(url, headers=headers, stream=True, verify=verify, timeout = 300)
            

    if not quiet:
        logger.info("Downloading started ...")
        if resume:
            logger.info("Resume: ", tmp_file)
        logger.info("From: "+ url_origin)
        logger.info(
            "To: "+ osp.abspath(output) if output_is_path else output,)

    try:
        total = res.headers.get("Content-Length")
        if total is not None:
            total = int(total)
        if not quiet:
            pbar = tqdm.tqdm(total=total, unit="B", unit_scale=True)
        t_start = time.time()
        for chunk in res.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)
            if not quiet:
                pbar.update(len(chunk))
            if speed is not None:
                elapsed_time_expected = 1.0 * pbar.n / speed
                elapsed_time = time.time() - t_start
                if elapsed_time < elapsed_time_expected:
                    time.sleep(elapsed_time_expected - elapsed_time)
        if not quiet:
            pbar.close()
        if tmp_file:
            f.close()
            shutil.move(tmp_file, output)
    except IOError as e:
        logger.error(e)
        return "Error"
    except requests.exceptions.RequestException as e:  # This is the correct syntax
            return "Error"
    except requests.exceptions.ConnectionError as e:
        logger.error("An error has occurred connectiong to download server")
        return "Error"
    finally:
        sess.close()

    return output
