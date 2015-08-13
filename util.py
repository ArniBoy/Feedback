#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Arne'

import os
import codecs

root = '/Users/Ceca/Arne/Data/Corpora/'
html_dict = {u'&quot;': u'"', u'&amp;': u'&', u'&lt;': u'<', u'&gt;': u'>',
             u'&OElig;': u'Œ', u'&oelig;': u'œ', u'&Scaron;': u'Š',
             u'&scaron;': u'š', u'&Yuml;': u'Ÿ', u'&circ;': u'ˆ',
             u'&tilde;': u'˜', u'&ndash;': u'–', u'&mdash;': u'—',
             u'&lsquo;': u'‘', u'&rsquo;': u'’', u'&sbquo;': u'‚',
             u'&ldquo;': u'“', u'&rdquo;': u'”', u'&bdquo;': u'„',
             u'&dagger;': u'†', u'&Dagger;': u'‡', u'&permil;': u'‰',
             u'&lsaquo;': u'‹', u'&rsaquo;': u'›', u'&euro;': u'€}'}

def make_dir(d):
    """
    Wrapper to safely make target directory
    :param d: Directory to make
    """
    if not os.path.exists(d):
        os.makedirs(d)


def get_columns(doc, columns, delim, only_if=-1):
    """
    Extract columns from target document, write them to cleaned tsv file
    :param doc: Target document
    :param columns: Columns to be extracted
    :param delim: Column delimiter
    :param only_if: Ignore rows with different column count
    """
    with open(os.path.splitext(doc)[0]+'_cleaned.tsv', 'w') as sink:
        for line in open(doc):
            data = line.strip().split(delim)
            if 0 <= only_if == len(data) or only_if < 0:
                sub_row = ''
                for column in columns:
                    sub_row += '%s\t' % data[column]
                sink.write(sub_row[:-1] + '\n')


def batch_file(doc, batch_size=10000):
    """
    Split file into batches
    :param doc: Target document
    :param batch_size: Desired maximum batch size
    """
    d = os.path.join(os.path.dirname(doc), 'batches')
    make_dir(d)
    batch = []
    count = 0
    for line_a in open(doc):
        if len(batch) < batch_size:
            batch.append(line_a)
        else:
            with open(os.path.join(d, 'batch_%i.tsv' % count), 'w') as sink:
                for line_b in batch:
                    sink.write(line_b)
            batch = []
            count += 1
    else:
        with open(os.path.join(d, 'batch_%i.tsv' % count), 'w') as sink:
            for line_b in batch:
                sink.write(line_b)


def fix_html(doc):
    """
    Repairs html sub-symbols to original state -- needs to happen after column split, since " is part of the substituted
    symbols.
    :param doc: Original document, gets overwritten
    """
    fixed = []
    for line in codecs.open(doc, encoding='utf-8'):
        for key, val in html_dict.items():
            line = line.replace(key, val)
        fixed.append(line)
    with codecs.open(doc, 'w', encoding='utf-8') as sink:
        for line in fixed:
            sink.write(line)
