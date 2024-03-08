---
title: "Python 모듈 상대경로로 import 하기"
date: 2021-01-18T15:34:30+09:00
categories:
  - python
tags:
  - python
  - setuptool
  - python module
  - importerror
---

## ImportError

예전에 상대 경로로 파이썬 모듈을 임포트 했더니 오류가 발생한 경험이 있었다.

`ImportError: attempted relative import with no known parent package`

## 원인

에러에서 나와있듯이 파이썬이 parent package를 알 수 없으므로 발생한 문제였다.

## 해결방법

파이썬에게 상위 패키지를 알려주어야 하는데 그 방법은 -m 인자를 주어 실행시키는 것이다. 이것은 해당 파이썬 파일을 모듈로서 실행한다는 의미이다. 참고로 이 명령어를 쓰기 위해선 당연히 해당 모듈을 설치한 상태여야 한다.

실행하고 싶은 모듈 이름이 module이고 그 상위 모듈이 parent이라면 `python3 -m parent.module`와 같이 실행하면 된다.

## vscode로 개발 시

vscode 환경에서는 파이썬 모듈을 실행/디버그 할 때 명령어를 칠 필요 없이 버튼만 누르면 알아서 실행 명령어를 vscode내부 터미널로 전달해준다. 하지만 기본적으로 vscode가 실행해주는 명령어는 파이썬 모듈이 아닌 파일을 실행해 주는 명령이므로 이것을 수정할 필요가 있다.

디버그/실행 설정은 .vscode/launch.json 파일에서 담당한다. 이 파일을 만들기 위해선 vscode 왼쪽 메뉴바의 run 탭(벌레와 삼각형이 그려진 아이콘)을 클릭하고 'create a launch.json file'을 클릭한다. 그리고 실행환경을 Module로 선택하면 .vscode 폴더에 launch.json 파일이 만들어진다.

launch.json 의 `"module"` 값을 (상위 모듈의 이름이 parent 일 때) `"parent.${fileBasenameNoExtension}"` 으로 바꾼다.

```json
// launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Module",
      "type": "python",
      "request": "launch",
      "module": "parent.${fileBasenameNoExtension}"
    }
  ]
}
```

이와 같이 변경하면 vscode로 디버깅할 때 모듈로 실행할 수 있게 된다.
