---
title: "priority_queue emplace() vs push()"
categories: [c++, algorithm]
tags: [c++, algorithm, data structure]
---
알고리즘 문제 풀다가 c++의 priority\_queue에 대해 공부하던 중 emplace()와 push()라는 두 함수의 차이에 대해 궁금해졌다.

다른 블로그들의 설명에 따르면 emplace는 priority\_queue에 값을 집어넣고 정렬해주는 함수라고 하는데, 그러면 push랑 다른게 없다는 말인가?

찾아보니 다음과 같은 차이가 존재했다.

>When we use push(), we create an object and then insert it into the priority\_queue. With emplace(), the object is constructed in-place and saves an unnecessary copy. Please see [emplace vs insert in C++ STL](https://www.geeksforgeeks.org/emplace-vs-insert-c-stl/) for details.

즉 priority_queue에 값을 넣는 역할은 push와 같지만, emplace를 사용하면 즉시 오브젝트가 constructed 되고 불필요한 복사가 방지된다는 차이가 있다.

쓰임에서 차이는 다음과 같다.

```cpp
#include<bits/stdc++.h> 
using namespace std; 
    
int main() 
{ 
    // pair<char, int>를 요소로 갖는 priority_queue를 선언
    priority_queue<pair<char, int>> pqueue; 
        
    // pair<char, int>의 오브젝트를 만들지 않고 바로 값을 push한다
    pqueue.emplace('a', 24); 
        
    // Below line would not compile
    // pqueue.push('b', 25);     
        
    // push를 사용할 때에는 오브젝트를 만들어 준 다음에 넣어야 한다. 이 과정에서 불필요한 복사가 일어난다.
    pqueue.push(make_pair('b', 25));     
        
    // printing the priority_queue
    while (!pqueue.empty()) { 
        pair<char, int> p =  pqueue.top(); 
        cout << p.first << " "
             << p.second << endl; 
        pqueue.pop(); 
    } 
    // 출력 결과
    // b 25
    // a 24
    return 0; 
} 
```

한마디로 emplace()는 copy()와 constructor가 합쳐진 형태라고 할 수 있다.

>참고
>
>[www.geeksforgeeks.org/priority\_queue-emplace-in-cpp-stl/](https://www.geeksforgeeks.org/priority_queue-emplace-in-cpp-stl/)
