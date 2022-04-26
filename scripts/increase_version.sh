#!/bin/bash

echo "previous version:"
cat VERSION
echo "new version:"
read version

echo "$version" > VERSION
git add VERSION
git commit -m "set version"
git push origin

git tag -a v"$version" -m "version $version"
git push origin v"$version"
