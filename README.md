# MyLanguageFinalProject
## Programming Languages Final Project
```
pip install colorama
pip install sly
```
### Unary Minus
Correctly implements unary minus(ensure variable is defined before or else will not evaluate with intended value)
```
print -(x+1)
y:= 5
print -(y+3)
```
### Basic Type Checking
Outputs an error with mismatched types and an operation
```
print(1==true)
print(1+hello)
print(hello+1)
print(1/0)
```
### Sorting Lists
Can be done with any length of list of integers
```
print(sort [1,6,3,8,9])
print(quicksort [5,3,8,9,1,6,3,2])
```
### Indexing Lists
Can be done with any length of list. Can also check if trying to access out of bounds and stops program if it is
```
y := [4,7,2,8,0]
print(y[3])
print(y[6])
```
### Length of List
Finds the length of a list if it is stored in a variable
```
y:=[1,5,3,2,7,9,4,6]
print(length(y))
```
### Sum of List
Sums up all the integers in a list(stored in a variable) and returns the total int
```
y:=[1,5,3,2,7,9,4,6]
print(sum(y))
```
### Reverse a List
Reverses a list stored in a variable
```
y:=[1,5,3,2,7,9,4,6]
print(reverse(y))
```
### Finding Head/Tail of a List
When list is stored in a variable, this is how you can check
```
y:=[1,5,3,2,7,9,4,6]
print(head . y)
print(tail . y)
```
### Finding Min and Max of a List
Finds the min and max of a list when stored in a variable
```
y:=[1,5,3,2,7,9,4,6]
print(min(y))
print(max(y))
```
