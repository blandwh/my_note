# Introduction

一个莫名其妙的定义：“System One Model”。模型的能力：Send state and typed questions; get structured answers your code can use directly.

state 指的是判断所依据的信息，typed questions 规定了回答形式的问题。

引入 Jev 的原因，我感觉是 LLM 和一些场景的不匹配，也就是语言的局限，比如普通 LLM 经常输出一段供人阅读的文字，但程序需要的是一个明确的值。Jev 的输出是一个 “structured results”，可以直接使用，不需要像之前的 LLM 的输出加一个语言头做解析。

给了三种 primitives，分别是 Choice、Score 和 Noul。 

----------------------------------------------

貌似对之后的研究没用，先搁置了
