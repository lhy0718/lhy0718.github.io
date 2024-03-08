---
title: "Python Setuptool"
date: 2021-01-18T15:34:30+09:00
categories:
  - python
tags:
  - python
  - setuptool
  - python package
---

## Setuptool?

[Setuptool 공식 문서](https://setuptools.readthedocs.io/en/latest/index.html)

Setuptool이란 파이썬 패키징을 쉽게하기 위한 파이썬 라이브러리이다.

파이썬 패키지를 하나 만들어 보고 있는데, 관련 자료를 찾아보니 Setuptool 이라는게 있었다. 배포한 패키지를 쉽게 설치할 수 있게 하는 툴 인것 같다.

## Setuptool으로 무엇을 할 수 있을까?

- 파이썬 패키지와 모듈의 정의
- 배포 패키지의 메타데이터 포함
- 프로젝트 설치
- 등등

## Setuptool 의 설치

터미널에 입력 (프로젝트 루트 폴더에서):

`pip install --upgrade setuptools`

## 간단한 사용 예시

setup.py를 작성하여 패키지의 root 폴더에 넣는다.

```python
# setup.py
from setuptools import setup, find_packages
setup( #여기에 옵션들이 들어간다. 옵션들은 setup.cfg 파일로 분리할 수도 있다.
    name="HelloWorld",
    version="0.1",
    packages=find_packages(),
)
```

패키지를 설치하는 명령어는 다음과 같다. (프로젝트 루트 폴더에서)

`python setup.py install`

단, 개발환경 이라면,

`python setup.py develop`

이다.

install 옵션으로 설치하면 코드를 수정했을 때 다시 `python setup.py install`을 실행해야 한다. 그러므로 패키지를 개발중이라 코드를 자주 수정하고 테스트하는 상황이라면 `python setup.py develop`을 실행한다. develop옵션은 소스에 대한 링크를 생성하여 코드의 변화가 실시간으로 적용된다.

## setup.cfg

위의 예제에서 보면 setup()함수에 여러 설치 옵션들이 들어감을 알 수 있다. 이 옵션들은 setup.cfg이라는 별도의 파일에 따로 정리할 수 있다.

위의 예제에서 옵션을 setup.cfg파일로 뺀다면 다음과 같을 것 이다.

```cfg
# setup.cfg
[metadata]
name = HelloWorld
version = 0.1
[options]
packages = find:
```

```python
# setup.py
from setuptools import setup
setup()
```

setup.py가 훨씬 간단해진다.

[여기](https://setuptools.readthedocs.io/en/latest/setuptools.html#configuring-setup-using-setup-cfg-files)에 잘 정리 되어 있으니 참고하면 좋다.

setup.cfg에서 중요하다 생각되는 부분은 이것 이다. 프로젝트(패키지)의 디렉토리 구조를 Setuptools이 인식할 수 있도록 \[option\]에 알맞은 값을 넣어야 한다.

예를 들어 파일이 다음과 같은 구조일때

```txt
├── src
│   └── mypackage
│       ├── __init__.py
│       └── mod1.py
├── setup.py
└── setup.cfg
```

setup.cfg의 option부분은 이렇게 써야 한다.

```cfg
##...
[options]
package_dir = = src
packages = find:

[options.packages.find]
where = src
##...
```

이렇게 하지 않으면 패키지를 설치해도 사용할 수 없다.  
`ModuleNotFoundError: No module named ...` 가 뜰것이다.

## 패키지 배포

[링크 : packaging.python.org의 튜토리얼](https://packaging.python.org/tutorials/packaging-projects/#generating-distribution-archives)

## 생각해볼 점

setup.cfg, setup.py를 자동으로 생성해주거나 생성을 도와주는 별도의 프로그램이 있으면 좋겠다.
