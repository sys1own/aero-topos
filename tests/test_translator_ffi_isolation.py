# -*- coding: utf-8 -*-
"""Unit tests for translator.ffi_isolation."""

import unittest

from translator.ffi_isolation import (
    StackValue,
    AeroVMStack,
    python_to_stack,
    stack_to_python,
    FFICallResult,
    execute_ffi_call,
    FFIBatchResult,
    execute_ffi_batch,
    drain_stack,
)


class TestStackValue(unittest.TestCase):
    def test_default_ref_count(self):
        sv = StackValue(vtype="int", data=42)
        self.assertEqual(sv.ref_count, 1)


class TestAeroVMStack(unittest.TestCase):
    def test_push_and_pop(self):
        stack = AeroVMStack()
        sv = StackValue(vtype="int", data=10)
        stack.push(sv)
        self.assertEqual(stack.depth, 1)
        popped = stack.pop()
        self.assertEqual(popped.data, 10)
        self.assertEqual(stack.depth, 0)

    def test_pop_empty(self):
        stack = AeroVMStack()
        self.assertIsNone(stack.pop())

    def test_peek(self):
        stack = AeroVMStack()
        self.assertIsNone(stack.peek())
        sv = StackValue(vtype="string", data="hello")
        stack.push(sv)
        self.assertEqual(stack.peek().data, "hello")
        self.assertEqual(stack.depth, 1)

    def test_clear(self):
        stack = AeroVMStack()
        stack.push(StackValue(vtype="int", data=1))
        stack.push(StackValue(vtype="int", data=2))
        stack.clear()
        self.assertEqual(stack.depth, 0)
        self.assertEqual(stack._allocated, [])


class TestPythonToStack(unittest.TestCase):
    def test_none(self):
        sv = python_to_stack(None)
        self.assertEqual(sv.vtype, "null")
        self.assertIsNone(sv.data)

    def test_int(self):
        sv = python_to_stack(42)
        self.assertEqual(sv.vtype, "int")
        self.assertEqual(sv.data, 42)

    def test_float(self):
        sv = python_to_stack(3.14)
        self.assertEqual(sv.vtype, "float")
        self.assertAlmostEqual(sv.data, 3.14)

    def test_string(self):
        sv = python_to_stack("hello")
        self.assertEqual(sv.vtype, "string")
        self.assertEqual(sv.data, "hello")

    def test_bool(self):
        sv = python_to_stack(True)
        self.assertEqual(sv.vtype, "int")
        self.assertEqual(sv.data, 1)

    def test_list(self):
        sv = python_to_stack([1, 2, 3])
        self.assertEqual(sv.vtype, "array")
        self.assertEqual(sv.data, [1, 2, 3])

    def test_tuple(self):
        sv = python_to_stack((4, 5))
        self.assertEqual(sv.vtype, "array")
        self.assertEqual(sv.data, [4, 5])

    def test_dict_serialized(self):
        sv = python_to_stack({"key": "val"})
        self.assertEqual(sv.vtype, "string")
        self.assertIn("key", sv.data)


class TestStackToPython(unittest.TestCase):
    def test_null(self):
        sv = StackValue(vtype="null", data=None)
        self.assertIsNone(stack_to_python(sv))

    def test_int(self):
        sv = StackValue(vtype="int", data=99)
        self.assertEqual(stack_to_python(sv), 99)

    def test_array(self):
        sv = StackValue(vtype="array", data=[1, 2])
        self.assertEqual(stack_to_python(sv), [1, 2])


class TestExecuteFFICall(unittest.TestCase):
    def test_successful_call(self):
        stack = AeroVMStack()
        args = [python_to_stack(3), python_to_stack(4)]
        result = execute_ffi_call(lambda a, b: a + b, args, stack)
        self.assertTrue(result.success)
        self.assertEqual(result.return_value.data, 7)
        self.assertIsNone(result.error)
        self.assertEqual(stack.depth, 1)

    def test_call_with_exception(self):
        stack = AeroVMStack()
        args = [python_to_stack(1)]

        def bad_func(x):
            raise ValueError("test error")

        result = execute_ffi_call(bad_func, args, stack)
        self.assertFalse(result.success)
        self.assertIn("ValueError", result.error)
        self.assertEqual(stack.depth, 1)  # error pushed

    def test_deep_copy_isolation(self):
        stack = AeroVMStack()
        original_list = [1, 2, 3]
        args = [python_to_stack(original_list)]

        def mutate_list(lst):
            lst.append(999)
            return lst

        result = execute_ffi_call(mutate_list, args, stack)
        self.assertTrue(result.success)
        # Original list should not be mutated
        self.assertEqual(original_list, [1, 2, 3])


class TestExecuteFFIBatch(unittest.TestCase):
    def test_batch_execution(self):
        calls = [
            (lambda x: x * 2, [5]),
            (lambda x, y: x + y, [3, 4]),
        ]
        result = execute_ffi_batch(calls)
        self.assertIsInstance(result, FFIBatchResult)
        self.assertEqual(result.total_calls, 2)
        self.assertEqual(result.successful, 2)
        self.assertEqual(result.failed, 0)
        self.assertTrue(result.stack_clean)

    def test_batch_with_failure(self):
        def failing(x):
            raise RuntimeError("oops")

        calls = [
            (lambda x: x, [1]),
            (failing, [2]),
        ]
        result = execute_ffi_batch(calls)
        self.assertEqual(result.successful, 1)
        self.assertEqual(result.failed, 1)

    def test_batch_empty(self):
        result = execute_ffi_batch([])
        self.assertEqual(result.total_calls, 0)
        self.assertEqual(result.successful, 0)


class TestDrainStack(unittest.TestCase):
    def test_drain(self):
        stack = AeroVMStack()
        stack.push(StackValue(vtype="int", data=10))
        stack.push(StackValue(vtype="string", data="hi"))
        values = drain_stack(stack)
        self.assertEqual(values, ["hi", 10])
        self.assertEqual(stack.depth, 0)

    def test_drain_empty(self):
        stack = AeroVMStack()
        values = drain_stack(stack)
        self.assertEqual(values, [])


if __name__ == "__main__":
    unittest.main()
