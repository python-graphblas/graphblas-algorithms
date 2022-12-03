#!/usr/bin/env python
import argparse
import functools
import gzip
import io
import os
import sys
import tarfile

import requests

thisdir = os.path.dirname(__file__)
datapath = functools.partial(os.path.join, thisdir, "..", "data")

data_urls = {
    "amazon0302": "https://sparse.tamu.edu/MM/SNAP/amazon0302.tar.gz",
    "web-Google": "https://sparse.tamu.edu/MM/SNAP/web-Google.tar.gz",
    "soc-Pokec": "https://sparse.tamu.edu/MM/SNAP/soc-Pokec.tar.gz",
    "email-Enron": "https://sparse.tamu.edu/MM/SNAP/email-Enron.tar.gz",
    "preferentialAttachment": "https://sparse.tamu.edu/MM/DIMACS10/preferentialAttachment.tar.gz",
    "caidaRouterLevel": "https://sparse.tamu.edu/MM/DIMACS10/caidaRouterLevel.tar.gz",
    "dblp-2010": "https://sparse.tamu.edu/MM/LAW/dblp-2010.tar.gz",
    "citationCiteseer": "https://sparse.tamu.edu/MM/DIMACS10/citationCiteseer.tar.gz",
    "coAuthorsDBLP": "https://sparse.tamu.edu/MM/DIMACS10/coAuthorsDBLP.tar.gz",
    "as-Skitter": "https://sparse.tamu.edu/MM/SNAP/as-Skitter.tar.gz",
    "coPapersCiteseer": "https://sparse.tamu.edu/MM/DIMACS10/coPapersCiteseer.tar.gz",
    "coPapersDBLP": "https://sparse.tamu.edu/MM/DIMACS10/coPapersDBLP.tar.gz",
}


def download(url, target=None):
    req = requests.request("GET", url)
    assert req.ok, req.reason
    tar = tarfile.open(fileobj=io.BytesIO(gzip.decompress(req.content)))
    for member in tar.members:
        dirname, basename = os.path.split(member.name)
        if not basename.endswith(".mtx"):
            continue
        tar.extract(member)
        if target:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            os.replace(member.name, target)
            os.removedirs(dirname)


def main(datanames, overwrite=False):
    filenames = []
    for name in datanames:
        target = datapath(f"{name}.mtx")
        filenames.append(target)
        relpath = os.path.relpath(target)
        if not overwrite and os.path.exists(target):
            print(f"{relpath} already exists; skipping", file=sys.stderr)
            continue
        url = data_urls[name]
        print(f"Downloading {relpath} from {url}", file=sys.stderr)
        download(url, target)
    return filenames


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datanames", nargs="*", choices=list(data_urls) + [[]])
    args = parser.parse_args()
    datanames = args.datanames
    if not datanames:
        # None specified, so download all that are missing
        datanames = data_urls
        overwrite = False
    else:
        overwrite = True
    main(datanames, overwrite=overwrite)
