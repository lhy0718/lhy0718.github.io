---
title: "Mac Xcode에서 openCV 사용하기"
date: 2020-09-20T15:34:30+09:00
categories:
  - c++
tags:
  - c++
  - mac
  - xcode
  - opencv
---

Xcode 에서 openCV를 사용하기 위해 [dgrld.tistory.com/34](https://dgrld.tistory.com/34) 이 블로그의 내용을 따라서 진행했다.

그런데 위의 내용을 똑같이 따라해도 프로젝트를 실행시키면 아래와 같은 에러가 발생했다.

```txt
dyld: Library not loaded: (...) Reason: no suitable image found. (...)
```

## 해결방법

해결방법은 [stackoverflow.com/questions/26978806/dyld-library-not-loaded-lib-libopencv-core-3-0-dylib-reason-image-not-found](https://stackoverflow.com/questions/26978806/dyld-library-not-loaded-lib-libopencv-core-3-0-dylib-reason-image-not-found) 여기서 찾았다.

> \[YourProjectFile\] --> \[YourTargetFile\] --> "Signing & Capabilities" --> and Enable "Disable Library Validation"

프로젝트 TARGETS 설정에서 **Signing & Capabilities**의 **Disable Library Validation** 체크박스를 선택하면 해결되는 문제였다.
