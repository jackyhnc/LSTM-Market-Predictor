

/* purpose: computes the dot product
 * parameters: a is a vector of size n. b is a vector of size n.
 * return: integer dot product of a and b
 */
int dotProduct(int *a, int *b, int length) {
  int dot_product_sum = 0;

  for (int i = 0; i < length; i++) {
    dot_product_sum += a[i] * b[i];
  }

  return dot_product_sum;
}





/* purpose: creates integer stack by allocating memory of length n
 * parameters: n is the length of the stack
 * return: returns the pointer to the new stack or null if malloc fails
 */
int *createStack(int n) {
  int *stack = malloc(n*sizeof(int));

  if (!stack) {
    printf("Malloc failed");
    return NULL;
  }

  return stack;
}

/* purpose: pushes element to top of stack
 * parameters: p is the value to insert, n is the number of elements, max_size is the maximum size of the stack, stack is the pointer to the stack array
 * return: pointer to the updated stack or null if realloc fails
 */
int *insertElement(int p, int *n, int *maxSize, int *stack) {
  if (*n >= *maxSize) {
    (*maxSize)++;
    
    int *new_stack = realloc(stack, *maxSize * sizeof(int));
    if (!new_stack) {
      printf("Reallocation failed\n");
      return NULL;
    }
   
    stack = new_stack;
  }
  
  stack[*n] = p;
  (*n)++;

  return stack;
}

/* purpose: removes element from top of stack
 * parameters: stack is pointer to stack array, n is the number of elements
 * return: void
 */
void removeElement(int *stack, int *n) {
  if (*n > 0) {
    (*n)--;
  }
}

/*
 * purpose: returns the top of the stack
 * parameters: stack is pointer to stack array, n is the number of elements
 * return: the value of the top element of the stack
 */
int top(int *stack, int n) {
  if (n > 0) {
    return stack[n - 1];
    
  }
}