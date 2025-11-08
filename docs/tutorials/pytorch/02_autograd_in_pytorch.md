# PyTorch Autograd: Руководство

## Основные концепции

### 1. Что такое Autograd?

**Autograd** — это встроенный дифференциальный движок PyTorch, который автоматически вычисляет градиенты для любого вычислительного графа. Это неотъемлемая составляющая, необходимая при обучении нейронных сетей.

### 2. Вычислительный граф (Computational Graph)

Вычислительный граф — это направленный ациклический граф (DAG), где:
- **Листья** — входные тензоры (данные и параметры)
- **Корни** — выходные тензоры (loss, predictions)
- **Рёбра** — операции между тензорами

Каждая операция создаёт объект класса `Function`, который содержит:
- Функцию для **forward pass** (прямого прохода)
- Функцию для **backward pass** (обратного прохода) — вычисление производных

### 3. Требуемые градиенты: `requires_grad`

```python
# Способ 1: При создании тензора
w = torch.randn(5, 3, requires_grad=True)

# Способ 2: Изменение существующего тензора
x = torch.ones(5)
x.requires_grad_(True)
```

Только для тензоров с `requires_grad=True` будут вычисляться градиенты. Это нужно устанавливать для параметров модели (весов и смещений), которые мы хотим оптимизировать. 

### 4. Как работает Forward Pass?

Когда вычисляется выражение вида `z = torch.matmul(x, w) + b`:
1. **Запускается операция** — результат сохраняется в новый тензор
2. **Сохраняется граф** — в свойство `grad_fn` записывается функция, которая использовалась
  
```python
z = torch.matmul(x, w) + b
print(z.grad_fn)  # <AddBackward0 object at 0x...>

loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
print(loss.grad_fn)  # <BinaryCrossEntropyWithLogitsBackward0 object at 0x...>
```

### 5. Как работает Backward Pass?

Backward pass инициируется вызовом `.backward()` на корневом узле (обычно на loss):

```python
loss.backward()
```

Процесс:
1. **Вычисление локальных производных** — для каждого `grad_fn` вычисляется его производная
2. **Применение правила цепи** — производные перемножаются по цепи: $\frac{\partial loss}{\partial x} = \frac{\partial loss}{\partial z} \cdot \frac{\partial z}{\partial x}$
3. **Накопление градиентов** — результаты добавляются в атрибут `.grad` листовых узлов

Результат:
```python
print(w.grad)  # Градиент loss по w
print(b.grad)  # Градиент loss по b
```

### 6. Примечание

#### 6.1 DAGs динамичны

После каждого вызова `.backward()` граф пересоздаётся. Это позволяет использовать условные операторы и динамичные вычисления.

#### 6.2 Только листовые узлы получают градиенты

```python
# Это работает (w и b — листовые узлы)
print(w.grad)
print(b.grad)  

# Это не сработает (z не листовой узел)
print(z.grad)  # None
```

#### 6.3 Градиенты накапливаются

При каждом вызове `.backward()` новые градиенты добавляются к существующим:

```python
loss.backward()
print(w.grad)  # Первый набор градиентов
loss.backward()  # ОШИБКА! Без retain_graph=True

# Правильно:
loss.backward(retain_graph=True)
print(w.grad)  # Градиенты удвоены (накоплены)

# Очистка:
w.grad.zero_()
loss.backward()
print(w.grad)  # Новые градиенты
```

### 7. Отключение отслеживания градиентов

Есть случаи, когда градиенты не нужны:

```python
# Способ 1: Контекстный менеджер
with torch.no_grad():
    z = torch.matmul(x, w) + b  # requires_grad будет False

# Способ 2: Метод detach()
z_no_grad = z.detach()  # Новый тензор без отслеживания

# Способ 3: Для модели на тестировании
model.eval()
with torch.no_grad():
    predictions = model(test_data)
```

### 8. Дефолтный аргумент для `.backward()`

```python
loss.backward()
# Эквивалентно:
loss.backward(torch.tensor(1.0))
```
Это работает только для скалярных функций (например, loss при обучении).

## Практический пример: Обучение простой нейросети

```python
import torch

# 1. Инициализация данных и параметров
x = torch.ones(5)           # входные данные
y = torch.zeros(3)          # целевой выход
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

# 2. Forward pass
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# 3. Backward pass
loss.backward()

# 4. Обновление параметров (в реальности это делает оптимизатор)
learning_rate = 0.01
with torch.no_grad():
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad
    w.grad.zero_()
    b.grad.zero_()

# 5. Повторить для следующего
```

## Заключение

- **Autograd автоматизирует дифференцирование** — не нужно вручную писать формулы для производных
- **Вычислительный граф динамичен** — можно использовать условные операторы python’а
- **Только листовые узлы получают градиенты** — промежуточные значения не имеют `.grad`
- **Градиенты накапливаются** — нужно вызывать `.zero_()` перед новым backward pass
- **Можно отключить отслеживание** — это ускоряет вычисления на inference стадии
