# 样例

## 简介
这里写本教程的内容，之后出推送的时候可以直接把它复制到推送里。

## 文件夹组织
一般来说，一个C++教程单元中有以下三个子文件夹：
- <code>docs/</code>：用来存放<code>.md</code>教程文件
- <code>imgs/</code>：用来存放<code>docs/</code>中教程所需图片（注意路径）
- <code>codes/</code>：用来存放能运行的代码示例
  

各位在写这一块的时候，只要说明下各文件夹下有什么文件、有什么作用就行。

## 参考资料
咱们C++教程的主要参考来源于以下讲义、书籍与网站：
- 清华大学计算机系姚海龙老师程序设计基础课件
- 《C++ Primer》（第五版）
- [cplusplus reference](https://www.cplusplus.com/reference/)
- [learn C++](https://www.learncpp.com/)
- [Microsoft C++ docs](https://docs.microsoft.com/en-us/cpp/cpp/?view=msvc-170)
- [GeeksforGeeks C++ tutorials](https://www.geeksforgeeks.org/c-plus-plus/)

各位在写这部分的时候把自己参考的资料列出来就行。

---

下面是一些你们写<code>README.md</code>的时候不必出现的东西：

## 有关类图的绘制

第六至十讲可能要手画一些简单的UML类图，有关C++类间关系及类图的参考资料如下：
- [UML Class Diagram Explained With C++ samples](https://cppcodetips.wordpress.com/2013/12/23/uml-class-diagram-explained-with-c-samples/)
- [ C++ OOD and OOP - Class Diagram in UML](https://www.youtube.com/watch?v=thbxWbneJ6o)

这里推荐[PlantUML](http://www.plantuml.com/plantuml/uml/SyfFKj2rKt3CoKnELR1Io4ZDoSa70000)。有关如何使用PlantUML画UML类图的参考资料如下：
- [PlantUML类图介绍](https://plantuml.com/zh/class-diagram)

生成UML类图后，只需在<code>.md</code>文件中附上图片链接，并在链接前加上<code>//https:</code>，即可显示。简单示例如下：
![UML示例图](https://www.plantuml.com/plantuml/png/SyfFKj2rKt3CoKnELR1Io4ZDoSa70000)

## 有关<code>.md</code>文件中图片显示的问题

为了使得图片能够在非本地（如GitHub）能正常显示，你可能需要手动写下图片路径：

例如现在我处于<code>C++/样例/README.md</code>，想要去显示位于<code>MATLAB/快速入门/test_image/</code>中的<code>img1.jpg</code>，一个恰当的做法如下：
![example](../../MATLAB/快速入门/test_image/img1.jpg)

亦即，你需要把绝对路径改成相对路径。

```
int main() {
    return 0;
    int &
}
```