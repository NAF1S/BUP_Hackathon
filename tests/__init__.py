"""Test package marker.

Without this file pytest can resolve ``tests`` to an unrelated package inside
``site-packages``, which breaks intra-suite imports.
"""
