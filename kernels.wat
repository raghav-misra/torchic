(module
  (type (;0;) (func))
  (type (;1;) (func (param i32 i32 i32 i32 i32)))
  (type (;2;) (func (param i32 i32 i32 i32 i32 i32 i32 i32 i32)))
  (type (;3;) (func (param i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)))
  (type (;4;) (func (param i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)))
  (type (;5;) (func (param i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)))
  (type (;6;) (func (param i32 i32 i32 i32)))
  (type (;7;) (func (param f32) (result f32)))
  (type (;8;) (func (param i32 i32 i32 i32 i32 i32)))
  (type (;9;) (func (param i32 f32 i32 i32)))
  (type (;10;) (func (param i32 i32 f32 i32 i32)))
  (type (;11;) (func (param i32 i32 i32 f32 i32 i32)))
  (type (;12;) (func (param i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)))
  (type (;13;) (func (param i32 i32 i32 i32 i32 i32 i32)))
  (type (;14;) (func (param i32 i32 i32 i32 i32 f32 i32 i32)))
  (type (;15;) (func (param f64 i32 i32) (result i32)))
  (type (;16;) (func (param i32 i32 i32)))
  (import "env" "memory" (memory (;0;) 17 32768 shared))
  (global (;0;) (mut i32) i32.const 1048576)
  (global (;1;) (mut i32) i32.const 0)
  (export "__stack_pointer" (global 0))
  (export "add" (func 1))
  (export "add_broadcast" (func 2))
  (export "add_scalar_tensor" (func 4))
  (export "bmm" (func 5))
  (export "matmul" (func 6))
  (export "concat_slab" (func 7))
  (export "conv1d" (func 8))
  (export "conv_transpose1d" (func 9))
  (export "copy" (func 10))
  (export "cos" (func 11))
  (export "div" (func 13))
  (export "embedding" (func 14))
  (export "embedding_backward" (func 15))
  (export "exp" (func 16))
  (export "fill" (func 18))
  (export "gelu" (func 19))
  (export "gelu_backward" (func 21))
  (export "leaky_relu" (func 22))
  (export "leaky_relu_backward" (func 23))
  (export "log" (func 24))
  (export "lstm_step" (func 25))
  (export "materialize" (func 26))
  (export "mul" (func 27))
  (export "neg" (func 28))
  (export "randn" (func 29))
  (export "relu" (func 31))
  (export "relu_backward" (func 32))
  (export "rms_norm2d" (func 33))
  (export "rsqrt_backward" (func 34))
  (export "rsqrt_op" (func 35))
  (export "sigmoid" (func 36))
  (export "sigmoid_backward" (func 37))
  (export "silu" (func 38))
  (export "silu_backward" (func 39))
  (export "sin" (func 40))
  (export "softmax2d" (func 42))
  (export "softmax_backward2d" (func 43))
  (export "sqrt_backward" (func 44))
  (export "sqrt_op" (func 45))
  (export "sub" (func 46))
  (export "sum_axis" (func 47))
  (export "sum_final" (func 48))
  (export "sum_partial" (func 49))
  (export "tanh" (func 50))
  (export "tanh_backward" (func 51))
  (export "transpose" (func 52))
  (start 0)
  (func (;0;) (type 0)
    block ;; label = @1
      block ;; label = @2
        block ;; label = @3
          i32.const 1049104
          i32.const 0
          i32.const 1
          i32.atomic.rmw.cmpxchg
          br_table 0 (;@3;) 1 (;@2;) 2 (;@1;)
        end
        i32.const 1048576
        i32.const 0
        i32.const 528
        memory.init 0
        i32.const 1049104
        i32.const 2
        i32.atomic.store
        i32.const 1049104
        i32.const -1
        memory.atomic.notify
        drop
        br 1 (;@1;)
      end
      i32.const 1049104
      i32.const 1
      i64.const -1
      memory.atomic.wait32
      drop
    end
    data.drop 0
  )
  (func (;1;) (type 1) (param i32 i32 i32 i32 i32)
    (local i32 i32 i32 i32 i32 i32)
    block ;; label = @1
      local.get 3
      i32.const 4
      i32.add
      local.get 4
      i32.gt_u
      br_if 0 (;@1;)
      local.get 2
      local.get 3
      i32.const 2
      i32.shl
      local.tee 5
      i32.add
      local.set 6
      local.get 1
      local.get 5
      i32.add
      local.set 7
      local.get 0
      local.get 5
      i32.add
      local.set 5
      loop ;; label = @2
        local.get 6
        local.get 5
        v128.load align=1
        local.get 7
        v128.load align=1
        f32x4.add
        v128.store align=1
        local.get 6
        i32.const 16
        i32.add
        local.set 6
        local.get 7
        i32.const 16
        i32.add
        local.set 7
        local.get 5
        i32.const 16
        i32.add
        local.set 5
        local.get 3
        local.tee 8
        i32.const 4
        i32.add
        local.set 3
        local.get 8
        i32.const 8
        i32.add
        local.get 4
        i32.le_u
        br_if 0 (;@2;)
      end
    end
    block ;; label = @1
      local.get 4
      local.get 3
      i32.le_u
      br_if 0 (;@1;)
      block ;; label = @2
        local.get 4
        local.get 3
        i32.sub
        local.tee 9
        i32.const 8
        i32.lt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 2
        i32.sub
        i32.const -16
        i32.gt_u
        br_if 0 (;@2;)
        local.get 1
        local.get 2
        i32.sub
        i32.const -16
        i32.gt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 3
        i32.const 2
        i32.shl
        local.tee 5
        i32.add
        local.set 6
        local.get 1
        local.get 5
        i32.add
        local.set 7
        local.get 2
        local.get 5
        i32.add
        local.set 5
        local.get 3
        local.get 9
        i32.const -4
        i32.and
        local.tee 10
        i32.add
        local.set 3
        local.get 10
        local.set 8
        loop ;; label = @3
          local.get 5
          local.get 6
          v128.load align=4
          local.get 7
          v128.load align=4
          f32x4.add
          v128.store align=4
          local.get 6
          i32.const 16
          i32.add
          local.set 6
          local.get 7
          i32.const 16
          i32.add
          local.set 7
          local.get 5
          i32.const 16
          i32.add
          local.set 5
          local.get 8
          i32.const -4
          i32.add
          local.tee 8
          br_if 0 (;@3;)
        end
        local.get 9
        local.get 10
        i32.eq
        br_if 1 (;@1;)
      end
      local.get 3
      i32.const 1
      i32.add
      local.set 6
      block ;; label = @2
        local.get 4
        local.get 3
        i32.sub
        i32.const 1
        i32.and
        i32.eqz
        br_if 0 (;@2;)
        local.get 2
        local.get 3
        i32.const 2
        i32.shl
        local.tee 7
        i32.add
        local.get 0
        local.get 7
        i32.add
        f32.load
        local.get 1
        local.get 7
        i32.add
        f32.load
        f32.add
        f32.store
        local.get 6
        local.set 3
      end
      local.get 4
      local.get 6
      i32.eq
      br_if 0 (;@1;)
      local.get 4
      local.get 3
      i32.sub
      local.set 8
      local.get 0
      local.get 3
      i32.const 2
      i32.shl
      local.tee 5
      i32.add
      local.set 6
      local.get 1
      local.get 5
      i32.add
      local.set 7
      local.get 2
      local.get 5
      i32.add
      local.set 5
      loop ;; label = @2
        local.get 5
        local.get 6
        f32.load
        local.get 7
        f32.load
        f32.add
        f32.store
        local.get 5
        i32.const 4
        i32.add
        local.get 6
        i32.const 4
        i32.add
        f32.load
        local.get 7
        i32.const 4
        i32.add
        f32.load
        f32.add
        f32.store
        local.get 6
        i32.const 8
        i32.add
        local.set 6
        local.get 7
        i32.const 8
        i32.add
        local.set 7
        local.get 5
        i32.const 8
        i32.add
        local.set 5
        local.get 8
        i32.const -2
        i32.add
        local.tee 8
        br_if 0 (;@2;)
      end
    end
  )
  (func (;2;) (type 2) (param i32 i32 i32 i32 i32 i32 i32 i32 i32)
    (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 v128)
    block ;; label = @1
      block ;; label = @2
        local.get 4
        local.get 3
        i32.le_u
        br_if 0 (;@2;)
        block ;; label = @3
          local.get 5
          i32.eqz
          br_if 0 (;@3;)
          local.get 8
          local.get 5
          i32.const 2
          i32.shl
          i32.const -4
          i32.add
          local.tee 9
          i32.add
          local.set 10
          local.get 7
          local.get 9
          i32.add
          local.set 11
          local.get 6
          local.get 9
          i32.add
          local.set 12
          loop ;; label = @4
            local.get 3
            i32.const 1
            i32.add
            local.set 13
            i32.const 0
            local.set 14
            local.get 10
            local.set 7
            local.get 11
            local.set 6
            local.get 12
            local.set 8
            local.get 5
            local.set 15
            local.get 3
            local.set 9
            i32.const 0
            local.set 16
            loop ;; label = @5
              local.get 8
              i32.load
              local.tee 17
              i32.eqz
              br_if 4 (;@1;)
              local.get 8
              i32.const -4
              i32.add
              local.set 8
              local.get 7
              i32.load
              local.get 9
              local.get 9
              local.get 17
              i32.div_u
              local.tee 18
              local.get 17
              i32.mul
              i32.sub
              local.tee 9
              i32.mul
              local.get 16
              i32.add
              local.set 16
              local.get 6
              i32.load
              local.get 9
              i32.mul
              local.get 14
              i32.add
              local.set 14
              local.get 7
              i32.const -4
              i32.add
              local.set 7
              local.get 6
              i32.const -4
              i32.add
              local.set 6
              local.get 18
              local.set 9
              local.get 15
              i32.const -1
              i32.add
              local.tee 15
              br_if 0 (;@5;)
            end
            local.get 2
            local.get 3
            i32.const 2
            i32.shl
            i32.add
            local.get 0
            local.get 14
            i32.const 2
            i32.shl
            i32.add
            f32.load
            local.get 1
            local.get 16
            i32.const 2
            i32.shl
            i32.add
            f32.load
            f32.add
            f32.store
            local.get 13
            local.set 3
            local.get 13
            local.get 4
            i32.ne
            br_if 0 (;@4;)
            br 2 (;@2;)
          end
        end
        block ;; label = @3
          local.get 4
          local.get 3
          i32.sub
          local.tee 6
          i32.const 16
          i32.lt_u
          br_if 0 (;@3;)
          local.get 2
          local.get 3
          i32.const 2
          i32.shl
          i32.add
          local.tee 8
          local.get 0
          i32.const 4
          i32.add
          i32.lt_u
          local.get 0
          local.get 2
          local.get 4
          i32.const 2
          i32.shl
          i32.add
          local.tee 7
          i32.lt_u
          i32.and
          br_if 0 (;@3;)
          local.get 8
          local.get 1
          i32.const 4
          i32.add
          i32.lt_u
          local.get 1
          local.get 7
          i32.lt_u
          i32.and
          br_if 0 (;@3;)
          local.get 2
          local.get 3
          i32.const 2
          i32.shl
          i32.add
          local.set 8
          local.get 3
          local.get 6
          i32.const -4
          i32.and
          local.tee 9
          i32.add
          local.set 3
          local.get 0
          f32.load
          local.get 1
          f32.load
          f32.add
          f32x4.splat
          local.set 19
          local.get 9
          local.set 7
          loop ;; label = @4
            local.get 8
            local.get 19
            v128.store align=4
            local.get 8
            i32.const 16
            i32.add
            local.set 8
            local.get 7
            i32.const -4
            i32.add
            local.tee 7
            br_if 0 (;@4;)
          end
          local.get 6
          local.get 9
          i32.eq
          br_if 1 (;@2;)
        end
        local.get 3
        local.set 6
        block ;; label = @3
          local.get 4
          local.get 3
          i32.sub
          i32.const 3
          i32.and
          local.tee 7
          i32.eqz
          br_if 0 (;@3;)
          local.get 3
          local.get 7
          i32.add
          local.set 6
          local.get 2
          local.get 3
          i32.const 2
          i32.shl
          i32.add
          local.set 8
          loop ;; label = @4
            local.get 8
            local.get 0
            f32.load
            local.get 1
            f32.load
            f32.add
            f32.store
            local.get 8
            i32.const 4
            i32.add
            local.set 8
            local.get 7
            i32.const -1
            i32.add
            local.tee 7
            br_if 0 (;@4;)
          end
        end
        local.get 3
        local.get 4
        i32.sub
        i32.const -4
        i32.gt_u
        br_if 0 (;@2;)
        local.get 4
        local.get 6
        i32.sub
        local.set 7
        local.get 2
        local.get 6
        i32.const 2
        i32.shl
        i32.add
        local.set 8
        loop ;; label = @3
          local.get 8
          local.get 0
          f32.load
          local.get 1
          f32.load
          f32.add
          f32.store
          local.get 8
          i32.const 4
          i32.add
          local.get 0
          f32.load
          local.get 1
          f32.load
          f32.add
          f32.store
          local.get 8
          i32.const 8
          i32.add
          local.get 0
          f32.load
          local.get 1
          f32.load
          f32.add
          f32.store
          local.get 8
          i32.const 12
          i32.add
          local.get 0
          f32.load
          local.get 1
          f32.load
          f32.add
          f32.store
          local.get 8
          i32.const 16
          i32.add
          local.set 8
          local.get 7
          i32.const -4
          i32.add
          local.tee 7
          br_if 0 (;@3;)
        end
      end
      return
    end
    call 3
    unreachable
  )
  (func (;3;) (type 0)
    call 53
    unreachable
  )
  (func (;4;) (type 1) (param i32 i32 i32 i32 i32)
    (local f32 v128 i32 i32 i32 i32)
    local.get 1
    f32.load
    local.tee 5
    f32x4.splat
    local.set 6
    block ;; label = @1
      local.get 3
      i32.const 4
      i32.add
      local.get 4
      i32.gt_u
      br_if 0 (;@1;)
      local.get 2
      local.get 3
      i32.const 2
      i32.shl
      local.tee 7
      i32.add
      local.set 1
      local.get 0
      local.get 7
      i32.add
      local.set 7
      loop ;; label = @2
        local.get 1
        local.get 6
        local.get 7
        v128.load align=1
        f32x4.add
        v128.store align=1
        local.get 1
        i32.const 16
        i32.add
        local.set 1
        local.get 7
        i32.const 16
        i32.add
        local.set 7
        local.get 3
        local.tee 8
        i32.const 4
        i32.add
        local.set 3
        local.get 8
        i32.const 8
        i32.add
        local.get 4
        i32.le_u
        br_if 0 (;@2;)
      end
    end
    block ;; label = @1
      local.get 4
      local.get 3
      i32.le_u
      br_if 0 (;@1;)
      block ;; label = @2
        local.get 4
        local.get 3
        i32.sub
        local.tee 9
        i32.const 4
        i32.lt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 2
        i32.sub
        i32.const -16
        i32.gt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 3
        i32.const 2
        i32.shl
        local.tee 7
        i32.add
        local.set 1
        local.get 2
        local.get 7
        i32.add
        local.set 7
        local.get 3
        local.get 9
        i32.const -4
        i32.and
        local.tee 10
        i32.add
        local.set 3
        local.get 10
        local.set 8
        loop ;; label = @3
          local.get 7
          local.get 6
          local.get 1
          v128.load align=4
          f32x4.add
          v128.store align=4
          local.get 1
          i32.const 16
          i32.add
          local.set 1
          local.get 7
          i32.const 16
          i32.add
          local.set 7
          local.get 8
          i32.const -4
          i32.add
          local.tee 8
          br_if 0 (;@3;)
        end
        local.get 9
        local.get 10
        i32.eq
        br_if 1 (;@1;)
      end
      local.get 3
      local.set 9
      block ;; label = @2
        local.get 4
        local.get 3
        i32.sub
        i32.const 3
        i32.and
        local.tee 8
        i32.eqz
        br_if 0 (;@2;)
        local.get 3
        local.get 8
        i32.add
        local.set 9
        local.get 0
        local.get 3
        i32.const 2
        i32.shl
        local.tee 7
        i32.add
        local.set 1
        local.get 2
        local.get 7
        i32.add
        local.set 7
        loop ;; label = @3
          local.get 7
          local.get 5
          local.get 1
          f32.load
          f32.add
          f32.store
          local.get 1
          i32.const 4
          i32.add
          local.set 1
          local.get 7
          i32.const 4
          i32.add
          local.set 7
          local.get 8
          i32.const -1
          i32.add
          local.tee 8
          br_if 0 (;@3;)
        end
      end
      local.get 3
      local.get 4
      i32.sub
      i32.const -4
      i32.gt_u
      br_if 0 (;@1;)
      local.get 9
      i32.const 2
      i32.shl
      local.set 3
      local.get 4
      local.get 9
      i32.sub
      local.set 8
      loop ;; label = @2
        local.get 2
        local.get 3
        i32.add
        local.tee 1
        local.get 5
        local.get 0
        local.get 3
        i32.add
        local.tee 7
        f32.load
        f32.add
        f32.store
        local.get 1
        i32.const 4
        i32.add
        local.get 5
        local.get 7
        i32.const 4
        i32.add
        f32.load
        f32.add
        f32.store
        local.get 1
        i32.const 8
        i32.add
        local.get 5
        local.get 7
        i32.const 8
        i32.add
        f32.load
        f32.add
        f32.store
        local.get 1
        i32.const 12
        i32.add
        local.get 5
        local.get 7
        i32.const 12
        i32.add
        f32.load
        f32.add
        f32.store
        local.get 0
        i32.const 16
        i32.add
        local.set 0
        local.get 2
        i32.const 16
        i32.add
        local.set 2
        local.get 8
        i32.const -4
        i32.add
        local.tee 8
        br_if 0 (;@2;)
      end
    end
  )
  (func (;5;) (type 3) (param i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
    (local i32 i32 i32 i32 i32 i32)
    block ;; label = @1
      local.get 8
      local.get 7
      i32.le_u
      br_if 0 (;@1;)
      local.get 8
      local.get 7
      i32.sub
      local.set 10
      local.get 6
      local.get 4
      i32.mul
      i32.const 2
      i32.shl
      local.set 11
      local.get 6
      local.get 5
      i32.mul
      i32.const 2
      i32.shl
      local.set 12
      local.get 5
      local.get 4
      i32.mul
      local.tee 13
      i32.const 2
      i32.shl
      local.set 14
      local.get 0
      local.get 7
      local.get 6
      i32.mul
      local.tee 15
      local.get 4
      i32.mul
      i32.const 2
      i32.shl
      i32.add
      local.set 8
      local.get 1
      local.get 15
      local.get 5
      i32.mul
      i32.const 2
      i32.shl
      i32.add
      local.set 1
      local.get 2
      local.get 13
      local.get 7
      i32.mul
      i32.const 2
      i32.shl
      i32.add
      local.set 7
      loop ;; label = @2
        local.get 8
        local.get 1
        local.get 7
        local.get 8
        local.get 5
        local.get 6
        i32.const 0
        local.get 4
        local.get 6
        i32.const 1
        local.get 5
        i32.const 1
        local.get 9
        call 6
        local.get 8
        local.get 11
        i32.add
        local.set 8
        local.get 1
        local.get 12
        i32.add
        local.set 1
        local.get 7
        local.get 14
        i32.add
        local.set 7
        local.get 10
        i32.const -1
        i32.add
        local.tee 10
        br_if 0 (;@2;)
      end
    end
  )
  (func (;6;) (type 4) (param i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
    (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 v128 v128 v128 v128 v128 v128 v128 v128 v128 v128 v128 v128 v128 v128 f32)
    block ;; label = @1
      local.get 6
      i32.const 4
      i32.add
      local.tee 13
      local.get 7
      i32.gt_u
      br_if 0 (;@1;)
      block ;; label = @2
        local.get 4
        i32.const -8
        i32.and
        i32.const 0
        local.get 11
        i32.const 1
        i32.eq
        select
        local.tee 14
        i32.eqz
        br_if 0 (;@2;)
        local.get 4
        i32.const 2
        i32.shl
        local.set 15
        local.get 4
        i32.const 12
        i32.mul
        local.set 16
        local.get 4
        i32.const 3
        i32.shl
        local.set 17
        local.get 4
        local.get 14
        i32.sub
        i32.const 2
        i32.shl
        local.set 18
        local.get 10
        i32.const 3
        i32.shl
        local.set 19
        local.get 11
        i32.const 2
        i32.shl
        local.set 20
        local.get 9
        i32.const 3
        i32.shl
        local.set 21
        local.get 10
        i32.const 2
        i32.shl
        local.set 22
        local.get 10
        i32.const 10
        i32.shl
        local.set 23
        local.get 9
        i32.const 2
        i32.shl
        local.set 24
        local.get 8
        i32.const 4
        i32.shl
        local.set 25
        local.get 5
        i32.const -2
        i32.and
        local.set 26
        local.get 5
        i32.const 1
        i32.and
        local.set 27
        local.get 5
        i32.const -1
        i32.add
        local.set 28
        local.get 1
        local.get 14
        local.get 11
        i32.mul
        i32.const 2
        i32.shl
        i32.add
        local.set 29
        local.get 0
        local.get 8
        local.get 6
        i32.mul
        i32.const 2
        i32.shl
        i32.add
        local.tee 30
        local.set 31
        local.get 0
        local.get 8
        local.get 6
        i32.const 1
        i32.add
        i32.mul
        i32.const 2
        i32.shl
        i32.add
        local.tee 32
        local.set 33
        local.get 0
        local.get 8
        local.get 6
        i32.const 2
        i32.add
        i32.mul
        i32.const 2
        i32.shl
        i32.add
        local.tee 34
        local.set 35
        local.get 0
        local.get 8
        local.get 6
        i32.const 3
        i32.add
        i32.mul
        i32.const 2
        i32.shl
        i32.add
        local.tee 36
        local.set 37
        i32.const 0
        local.set 38
        loop ;; label = @3
          local.get 6
          local.set 39
          local.get 13
          local.set 6
          block ;; label = @4
            block ;; label = @5
              block ;; label = @6
                local.get 5
                i32.eqz
                br_if 0 (;@6;)
                local.get 36
                local.get 25
                local.get 38
                i32.mul
                local.tee 13
                i32.add
                local.set 40
                local.get 34
                local.get 13
                i32.add
                local.set 41
                local.get 32
                local.get 13
                i32.add
                local.set 42
                local.get 30
                local.get 13
                i32.add
                local.set 43
                local.get 2
                local.get 39
                local.get 4
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                local.set 44
                local.get 0
                local.get 39
                local.get 8
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                local.set 45
                local.get 0
                local.get 39
                i32.const 3
                i32.add
                local.tee 46
                local.get 8
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                local.set 47
                local.get 0
                local.get 39
                i32.const 2
                i32.add
                local.tee 48
                local.get 8
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                local.set 49
                local.get 0
                local.get 39
                i32.const 1
                i32.add
                local.tee 50
                local.get 8
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                local.set 51
                i32.const 0
                local.set 52
                local.get 1
                local.set 53
                local.get 31
                local.set 54
                local.get 33
                local.set 55
                local.get 35
                local.set 56
                local.get 37
                local.set 57
                local.get 5
                local.set 58
                i32.const 0
                local.set 59
                loop ;; label = @7
                  local.get 58
                  i32.const 1
                  local.get 58
                  i32.const 1
                  i32.gt_u
                  select
                  local.tee 60
                  i32.const 256
                  i32.lt_u
                  local.set 61
                  local.get 58
                  i32.const 256
                  local.get 58
                  i32.const 256
                  i32.lt_u
                  select
                  local.set 62
                  i32.const 0
                  local.set 63
                  block ;; label = @8
                    block ;; label = @9
                      local.get 5
                      local.get 59
                      i32.sub
                      local.tee 64
                      i32.const 8
                      i32.lt_u
                      br_if 0 (;@9;)
                      i32.const 0
                      local.set 63
                      local.get 9
                      i32.const 1
                      i32.ne
                      br_if 0 (;@9;)
                      i32.const 0
                      local.set 63
                      local.get 12
                      local.get 40
                      local.get 52
                      i32.const 10
                      i32.shl
                      local.tee 13
                      i32.add
                      local.tee 65
                      local.get 62
                      i32.const 2
                      i32.shl
                      local.tee 66
                      i32.add
                      i32.lt_u
                      local.get 65
                      local.get 12
                      local.get 62
                      i32.const 4
                      i32.shl
                      i32.add
                      local.tee 67
                      i32.lt_u
                      i32.and
                      br_if 0 (;@9;)
                      i32.const 0
                      local.set 63
                      local.get 12
                      local.get 41
                      local.get 13
                      i32.add
                      local.tee 65
                      local.get 66
                      i32.add
                      i32.lt_u
                      local.get 65
                      local.get 67
                      i32.lt_u
                      i32.and
                      br_if 0 (;@9;)
                      i32.const 0
                      local.set 63
                      local.get 12
                      local.get 42
                      local.get 13
                      i32.add
                      local.tee 65
                      local.get 66
                      i32.add
                      i32.lt_u
                      local.get 65
                      local.get 67
                      i32.lt_u
                      i32.and
                      br_if 0 (;@9;)
                      i32.const 0
                      local.set 63
                      local.get 12
                      local.get 43
                      local.get 13
                      i32.add
                      local.tee 13
                      local.get 66
                      i32.add
                      i32.lt_u
                      local.get 13
                      local.get 67
                      i32.lt_u
                      i32.and
                      br_if 0 (;@9;)
                      local.get 62
                      i32.const 508
                      i32.and
                      local.set 68
                      local.get 64
                      i32.const 256
                      local.get 64
                      i32.const 256
                      i32.lt_u
                      select
                      local.tee 69
                      i32.const 508
                      i32.and
                      local.set 63
                      local.get 12
                      local.set 13
                      local.get 54
                      local.set 66
                      local.get 55
                      local.set 67
                      local.get 56
                      local.set 64
                      local.get 57
                      local.set 65
                      loop ;; label = @10
                        local.get 13
                        local.get 66
                        v128.load align=4
                        local.tee 70
                        local.get 67
                        v128.load align=4
                        local.tee 71
                        i8x16.shuffle 12 13 14 15 28 29 30 31 0 1 2 3 0 1 2 3
                        local.get 64
                        v128.load align=4
                        local.tee 72
                        local.get 65
                        v128.load align=4
                        local.tee 73
                        i8x16.shuffle 0 1 2 3 0 1 2 3 12 13 14 15 28 29 30 31
                        i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31
                        v128.store offset=48 align=4
                        local.get 13
                        local.get 70
                        local.get 71
                        i8x16.shuffle 8 9 10 11 24 25 26 27 0 1 2 3 0 1 2 3
                        local.get 72
                        local.get 73
                        i8x16.shuffle 0 1 2 3 0 1 2 3 8 9 10 11 24 25 26 27
                        i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31
                        v128.store offset=32 align=4
                        local.get 13
                        local.get 70
                        local.get 71
                        i8x16.shuffle 4 5 6 7 20 21 22 23 0 1 2 3 0 1 2 3
                        local.get 72
                        local.get 73
                        i8x16.shuffle 0 1 2 3 0 1 2 3 4 5 6 7 20 21 22 23
                        i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31
                        v128.store offset=16 align=4
                        local.get 13
                        local.get 70
                        local.get 71
                        i8x16.shuffle 0 1 2 3 16 17 18 19 0 1 2 3 0 1 2 3
                        local.get 72
                        local.get 73
                        i8x16.shuffle 0 1 2 3 0 1 2 3 0 1 2 3 16 17 18 19
                        i8x16.shuffle 0 1 2 3 4 5 6 7 24 25 26 27 28 29 30 31
                        v128.store align=4
                        local.get 13
                        i32.const 64
                        i32.add
                        local.set 13
                        local.get 66
                        i32.const 16
                        i32.add
                        local.set 66
                        local.get 67
                        i32.const 16
                        i32.add
                        local.set 67
                        local.get 64
                        i32.const 16
                        i32.add
                        local.set 64
                        local.get 65
                        i32.const 16
                        i32.add
                        local.set 65
                        local.get 68
                        i32.const -4
                        i32.add
                        local.tee 68
                        br_if 0 (;@10;)
                      end
                      local.get 69
                      local.get 63
                      i32.eq
                      br_if 1 (;@8;)
                    end
                    local.get 62
                    local.get 63
                    i32.sub
                    local.set 67
                    local.get 12
                    local.get 63
                    i32.const 4
                    i32.shl
                    i32.add
                    local.set 13
                    local.get 24
                    local.get 63
                    local.get 59
                    i32.add
                    i32.mul
                    local.set 66
                    loop ;; label = @9
                      local.get 13
                      local.get 31
                      local.get 66
                      i32.add
                      f32.load
                      f32.store
                      local.get 13
                      i32.const 4
                      i32.add
                      local.get 33
                      local.get 66
                      i32.add
                      f32.load
                      f32.store
                      local.get 13
                      i32.const 8
                      i32.add
                      local.get 35
                      local.get 66
                      i32.add
                      f32.load
                      f32.store
                      local.get 13
                      i32.const 12
                      i32.add
                      local.get 37
                      local.get 66
                      i32.add
                      f32.load
                      f32.store
                      local.get 13
                      i32.const 16
                      i32.add
                      local.set 13
                      local.get 66
                      local.get 24
                      i32.add
                      local.set 66
                      local.get 67
                      i32.const -1
                      i32.add
                      local.tee 67
                      br_if 0 (;@9;)
                    end
                  end
                  local.get 60
                  i32.const 256
                  local.get 61
                  select
                  local.set 63
                  i32.const 0
                  local.set 65
                  local.get 53
                  local.set 68
                  loop ;; label = @8
                    local.get 44
                    local.get 65
                    i32.const 2
                    i32.shl
                    i32.add
                    local.set 64
                    block ;; label = @9
                      block ;; label = @10
                        local.get 59
                        br_if 0 (;@10;)
                        v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
                        local.tee 72
                        local.set 73
                        local.get 72
                        local.set 74
                        local.get 72
                        local.set 75
                        local.get 72
                        local.set 76
                        local.get 72
                        local.set 77
                        local.get 72
                        local.set 78
                        local.get 72
                        local.set 79
                        br 1 (;@9;)
                      end
                      local.get 64
                      local.get 16
                      i32.add
                      local.tee 13
                      v128.load align=1
                      local.set 78
                      local.get 64
                      local.get 17
                      i32.add
                      local.tee 66
                      v128.load align=1
                      local.set 76
                      local.get 13
                      i32.const 16
                      i32.add
                      v128.load align=1
                      local.set 79
                      local.get 66
                      i32.const 16
                      i32.add
                      v128.load align=1
                      local.set 77
                      local.get 64
                      local.get 15
                      i32.add
                      local.tee 13
                      v128.load align=1
                      local.set 74
                      local.get 13
                      i32.const 16
                      i32.add
                      v128.load align=1
                      local.set 75
                      local.get 64
                      v128.load offset=16 align=1
                      local.set 73
                      local.get 64
                      v128.load align=1
                      local.set 72
                    end
                    local.get 68
                    local.set 66
                    local.get 12
                    local.set 13
                    local.get 63
                    local.set 67
                    loop ;; label = @9
                      local.get 79
                      local.get 66
                      i32.const 16
                      i32.add
                      v128.load align=1
                      local.tee 70
                      local.get 13
                      i32.const 12
                      i32.add
                      v128.load32_splat align=1
                      local.tee 80
                      f32x4.mul
                      f32x4.add
                      local.set 79
                      local.get 77
                      local.get 13
                      i32.const 8
                      i32.add
                      v128.load32_splat align=1
                      local.tee 81
                      local.get 70
                      f32x4.mul
                      f32x4.add
                      local.set 77
                      local.get 75
                      local.get 13
                      i32.const 4
                      i32.add
                      v128.load32_splat align=1
                      local.tee 82
                      local.get 70
                      f32x4.mul
                      f32x4.add
                      local.set 75
                      local.get 72
                      local.get 13
                      v128.load32_splat align=1
                      local.tee 83
                      local.get 66
                      v128.load align=1
                      local.tee 71
                      f32x4.mul
                      f32x4.add
                      local.set 72
                      local.get 78
                      local.get 71
                      local.get 80
                      f32x4.mul
                      f32x4.add
                      local.set 78
                      local.get 76
                      local.get 71
                      local.get 81
                      f32x4.mul
                      f32x4.add
                      local.set 76
                      local.get 74
                      local.get 82
                      local.get 71
                      f32x4.mul
                      f32x4.add
                      local.set 74
                      local.get 73
                      local.get 83
                      local.get 70
                      f32x4.mul
                      f32x4.add
                      local.set 73
                      local.get 66
                      local.get 22
                      i32.add
                      local.set 66
                      local.get 13
                      i32.const 16
                      i32.add
                      local.set 13
                      local.get 67
                      i32.const -1
                      i32.add
                      local.tee 67
                      br_if 0 (;@9;)
                    end
                    local.get 64
                    local.get 73
                    v128.store offset=16 align=1
                    local.get 64
                    local.get 72
                    v128.store align=1
                    local.get 64
                    local.get 15
                    i32.add
                    local.tee 13
                    local.get 74
                    v128.store align=1
                    local.get 13
                    i32.const 16
                    i32.add
                    local.get 75
                    v128.store align=1
                    local.get 64
                    local.get 17
                    i32.add
                    local.tee 13
                    i32.const 16
                    i32.add
                    local.get 77
                    v128.store align=1
                    local.get 13
                    local.get 76
                    v128.store align=1
                    local.get 64
                    local.get 16
                    i32.add
                    local.tee 13
                    i32.const 16
                    i32.add
                    local.get 79
                    v128.store align=1
                    local.get 13
                    local.get 78
                    v128.store align=1
                    local.get 68
                    i32.const 32
                    i32.add
                    local.set 68
                    local.get 65
                    i32.const 8
                    i32.add
                    local.tee 65
                    local.get 14
                    i32.lt_u
                    br_if 0 (;@8;)
                  end
                  local.get 53
                  local.get 23
                  i32.add
                  local.set 53
                  local.get 54
                  i32.const 1024
                  i32.add
                  local.set 54
                  local.get 55
                  i32.const 1024
                  i32.add
                  local.set 55
                  local.get 56
                  i32.const 1024
                  i32.add
                  local.set 56
                  local.get 57
                  i32.const 1024
                  i32.add
                  local.set 57
                  local.get 52
                  i32.const 1
                  i32.add
                  local.set 52
                  local.get 58
                  i32.const -256
                  i32.add
                  local.set 58
                  local.get 59
                  i32.const 256
                  i32.add
                  local.tee 59
                  local.get 5
                  i32.ge_u
                  br_if 2 (;@5;)
                  br 0 (;@7;)
                end
              end
              local.get 4
              local.get 14
              i32.le_u
              br_if 1 (;@4;)
              local.get 39
              i32.const -4
              i32.ge_u
              br_if 1 (;@4;)
              local.get 2
              local.get 39
              local.get 4
              i32.mul
              local.get 14
              i32.add
              i32.const 2
              i32.shl
              i32.add
              local.set 66
              block ;; label = @6
                local.get 18
                i32.eqz
                local.tee 13
                br_if 0 (;@6;)
                local.get 66
                i32.const 0
                local.get 18
                memory.fill
              end
              block ;; label = @6
                local.get 13
                br_if 0 (;@6;)
                local.get 66
                local.get 15
                i32.add
                i32.const 0
                local.get 18
                memory.fill
              end
              block ;; label = @6
                local.get 13
                br_if 0 (;@6;)
                local.get 66
                local.get 17
                i32.add
                i32.const 0
                local.get 18
                memory.fill
              end
              local.get 13
              br_if 1 (;@4;)
              local.get 66
              local.get 16
              i32.add
              i32.const 0
              local.get 18
              memory.fill
              br 1 (;@4;)
            end
            local.get 4
            local.get 14
            i32.le_u
            br_if 0 (;@4;)
            local.get 29
            local.set 65
            local.get 14
            local.set 64
            local.get 39
            i32.const -5
            i32.gt_u
            br_if 0 (;@4;)
            loop ;; label = @5
              block ;; label = @6
                block ;; label = @7
                  block ;; label = @8
                    local.get 28
                    br_if 0 (;@8;)
                    f32.const 0x0p+0 (;=0;)
                    local.set 84
                    i32.const 0
                    local.set 67
                    br 1 (;@7;)
                  end
                  f32.const 0x0p+0 (;=0;)
                  local.set 84
                  i32.const 0
                  local.set 67
                  local.get 65
                  local.set 13
                  local.get 31
                  local.set 66
                  loop ;; label = @8
                    local.get 84
                    local.get 66
                    f32.load
                    local.get 13
                    f32.load
                    f32.mul
                    f32.add
                    local.get 66
                    local.get 24
                    i32.add
                    f32.load
                    local.get 13
                    local.get 22
                    i32.add
                    f32.load
                    f32.mul
                    f32.add
                    local.set 84
                    local.get 13
                    local.get 19
                    i32.add
                    local.set 13
                    local.get 66
                    local.get 21
                    i32.add
                    local.set 66
                    local.get 26
                    local.get 67
                    i32.const 2
                    i32.add
                    local.tee 67
                    i32.ne
                    br_if 0 (;@8;)
                  end
                  local.get 27
                  i32.eqz
                  br_if 1 (;@6;)
                end
                local.get 84
                local.get 45
                local.get 67
                local.get 9
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                f32.load
                local.get 1
                local.get 64
                local.get 11
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                local.get 67
                local.get 10
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                f32.load
                f32.mul
                f32.add
                local.set 84
              end
              local.get 44
              local.get 64
              i32.const 2
              i32.shl
              i32.add
              local.get 84
              f32.store
              local.get 65
              local.get 20
              i32.add
              local.set 65
              local.get 64
              i32.const 1
              i32.add
              local.tee 64
              local.get 4
              i32.ne
              br_if 0 (;@5;)
            end
            local.get 2
            local.get 50
            local.get 4
            i32.mul
            i32.const 2
            i32.shl
            i32.add
            local.set 68
            local.get 29
            local.set 65
            local.get 14
            local.set 64
            loop ;; label = @5
              f32.const 0x0p+0 (;=0;)
              local.set 84
              i32.const 0
              local.set 67
              block ;; label = @6
                block ;; label = @7
                  local.get 28
                  i32.eqz
                  br_if 0 (;@7;)
                  local.get 65
                  local.set 13
                  local.get 33
                  local.set 66
                  loop ;; label = @8
                    local.get 84
                    local.get 66
                    f32.load
                    local.get 13
                    f32.load
                    f32.mul
                    f32.add
                    local.get 66
                    local.get 24
                    i32.add
                    f32.load
                    local.get 13
                    local.get 22
                    i32.add
                    f32.load
                    f32.mul
                    f32.add
                    local.set 84
                    local.get 13
                    local.get 19
                    i32.add
                    local.set 13
                    local.get 66
                    local.get 21
                    i32.add
                    local.set 66
                    local.get 26
                    local.get 67
                    i32.const 2
                    i32.add
                    local.tee 67
                    i32.ne
                    br_if 0 (;@8;)
                  end
                  local.get 27
                  i32.eqz
                  br_if 1 (;@6;)
                end
                local.get 84
                local.get 51
                local.get 67
                local.get 9
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                f32.load
                local.get 1
                local.get 64
                local.get 11
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                local.get 67
                local.get 10
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                f32.load
                f32.mul
                f32.add
                local.set 84
              end
              local.get 68
              local.get 64
              i32.const 2
              i32.shl
              i32.add
              local.get 84
              f32.store
              local.get 65
              local.get 20
              i32.add
              local.set 65
              local.get 64
              i32.const 1
              i32.add
              local.tee 64
              local.get 4
              i32.ne
              br_if 0 (;@5;)
            end
            local.get 2
            local.get 48
            local.get 4
            i32.mul
            i32.const 2
            i32.shl
            i32.add
            local.set 68
            local.get 29
            local.set 65
            local.get 14
            local.set 64
            loop ;; label = @5
              f32.const 0x0p+0 (;=0;)
              local.set 84
              i32.const 0
              local.set 67
              block ;; label = @6
                block ;; label = @7
                  local.get 28
                  i32.eqz
                  br_if 0 (;@7;)
                  local.get 65
                  local.set 13
                  local.get 35
                  local.set 66
                  loop ;; label = @8
                    local.get 84
                    local.get 66
                    f32.load
                    local.get 13
                    f32.load
                    f32.mul
                    f32.add
                    local.get 66
                    local.get 24
                    i32.add
                    f32.load
                    local.get 13
                    local.get 22
                    i32.add
                    f32.load
                    f32.mul
                    f32.add
                    local.set 84
                    local.get 13
                    local.get 19
                    i32.add
                    local.set 13
                    local.get 66
                    local.get 21
                    i32.add
                    local.set 66
                    local.get 26
                    local.get 67
                    i32.const 2
                    i32.add
                    local.tee 67
                    i32.ne
                    br_if 0 (;@8;)
                  end
                  local.get 27
                  i32.eqz
                  br_if 1 (;@6;)
                end
                local.get 84
                local.get 49
                local.get 67
                local.get 9
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                f32.load
                local.get 1
                local.get 64
                local.get 11
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                local.get 67
                local.get 10
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                f32.load
                f32.mul
                f32.add
                local.set 84
              end
              local.get 68
              local.get 64
              i32.const 2
              i32.shl
              i32.add
              local.get 84
              f32.store
              local.get 65
              local.get 20
              i32.add
              local.set 65
              local.get 64
              i32.const 1
              i32.add
              local.tee 64
              local.get 4
              i32.ne
              br_if 0 (;@5;)
            end
            local.get 2
            local.get 46
            local.get 4
            i32.mul
            i32.const 2
            i32.shl
            i32.add
            local.set 68
            local.get 29
            local.set 65
            local.get 14
            local.set 64
            loop ;; label = @5
              f32.const 0x0p+0 (;=0;)
              local.set 84
              i32.const 0
              local.set 67
              block ;; label = @6
                block ;; label = @7
                  local.get 28
                  i32.eqz
                  br_if 0 (;@7;)
                  local.get 65
                  local.set 13
                  local.get 37
                  local.set 66
                  loop ;; label = @8
                    local.get 84
                    local.get 66
                    f32.load
                    local.get 13
                    f32.load
                    f32.mul
                    f32.add
                    local.get 66
                    local.get 24
                    i32.add
                    f32.load
                    local.get 13
                    local.get 22
                    i32.add
                    f32.load
                    f32.mul
                    f32.add
                    local.set 84
                    local.get 13
                    local.get 19
                    i32.add
                    local.set 13
                    local.get 66
                    local.get 21
                    i32.add
                    local.set 66
                    local.get 26
                    local.get 67
                    i32.const 2
                    i32.add
                    local.tee 67
                    i32.ne
                    br_if 0 (;@8;)
                  end
                  local.get 27
                  i32.eqz
                  br_if 1 (;@6;)
                end
                local.get 84
                local.get 47
                local.get 67
                local.get 9
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                f32.load
                local.get 1
                local.get 64
                local.get 11
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                local.get 67
                local.get 10
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                f32.load
                f32.mul
                f32.add
                local.set 84
              end
              local.get 68
              local.get 64
              i32.const 2
              i32.shl
              i32.add
              local.get 84
              f32.store
              local.get 65
              local.get 20
              i32.add
              local.set 65
              local.get 64
              i32.const 1
              i32.add
              local.tee 64
              local.get 4
              i32.ne
              br_if 0 (;@5;)
            end
          end
          local.get 31
          local.get 25
          i32.add
          local.set 31
          local.get 33
          local.get 25
          i32.add
          local.set 33
          local.get 35
          local.get 25
          i32.add
          local.set 35
          local.get 37
          local.get 25
          i32.add
          local.set 37
          local.get 38
          i32.const 1
          i32.add
          local.set 38
          local.get 6
          i32.const 4
          i32.add
          local.tee 13
          local.get 7
          i32.le_u
          br_if 0 (;@3;)
          br 2 (;@1;)
        end
      end
      block ;; label = @2
        local.get 4
        local.get 14
        i32.gt_u
        br_if 0 (;@2;)
        loop ;; label = @3
          local.get 6
          local.tee 13
          i32.const 4
          i32.add
          local.set 6
          local.get 13
          i32.const 8
          i32.add
          local.get 7
          i32.le_u
          br_if 0 (;@3;)
          br 2 (;@1;)
        end
      end
      block ;; label = @2
        local.get 5
        i32.eqz
        br_if 0 (;@2;)
        local.get 10
        i32.const 2
        i32.shl
        local.set 67
        local.get 9
        i32.const 2
        i32.shl
        local.set 22
        local.get 10
        i32.const 3
        i32.shl
        local.set 64
        local.get 11
        i32.const 2
        i32.shl
        local.set 15
        local.get 9
        i32.const 3
        i32.shl
        local.set 65
        local.get 8
        i32.const 4
        i32.shl
        local.set 33
        local.get 5
        i32.const 1
        i32.and
        local.set 21
        local.get 5
        i32.const -2
        i32.and
        local.set 68
        local.get 5
        i32.const -1
        i32.add
        local.set 16
        local.get 0
        local.get 8
        local.get 6
        i32.mul
        i32.const 2
        i32.shl
        i32.add
        local.set 59
        local.get 0
        local.get 8
        local.get 6
        i32.const 3
        i32.add
        i32.mul
        i32.const 2
        i32.shl
        i32.add
        local.set 12
        local.get 0
        local.get 8
        local.get 6
        i32.const 2
        i32.add
        i32.mul
        i32.const 2
        i32.shl
        i32.add
        local.set 63
        local.get 0
        local.get 8
        local.get 6
        i32.const 1
        i32.add
        i32.mul
        i32.const 2
        i32.shl
        i32.add
        local.set 44
        local.get 6
        local.set 31
        loop ;; label = @3
          local.get 13
          local.set 6
          block ;; label = @4
            local.get 31
            i32.const -5
            i32.gt_u
            br_if 0 (;@4;)
            local.get 2
            local.get 31
            local.get 4
            i32.mul
            i32.const 2
            i32.shl
            i32.add
            local.set 26
            local.get 0
            local.get 31
            local.get 8
            i32.mul
            i32.const 2
            i32.shl
            i32.add
            local.set 14
            i32.const 0
            local.set 17
            local.get 1
            local.set 19
            loop ;; label = @5
              block ;; label = @6
                block ;; label = @7
                  block ;; label = @8
                    local.get 16
                    br_if 0 (;@8;)
                    f32.const 0x0p+0 (;=0;)
                    local.set 84
                    i32.const 0
                    local.set 24
                    br 1 (;@7;)
                  end
                  f32.const 0x0p+0 (;=0;)
                  local.set 84
                  i32.const 0
                  local.set 24
                  local.get 19
                  local.set 13
                  local.get 59
                  local.set 66
                  loop ;; label = @8
                    local.get 84
                    local.get 66
                    f32.load
                    local.get 13
                    f32.load
                    f32.mul
                    f32.add
                    local.get 66
                    local.get 22
                    i32.add
                    f32.load
                    local.get 13
                    local.get 67
                    i32.add
                    f32.load
                    f32.mul
                    f32.add
                    local.set 84
                    local.get 13
                    local.get 64
                    i32.add
                    local.set 13
                    local.get 66
                    local.get 65
                    i32.add
                    local.set 66
                    local.get 68
                    local.get 24
                    i32.const 2
                    i32.add
                    local.tee 24
                    i32.ne
                    br_if 0 (;@8;)
                  end
                  local.get 21
                  i32.eqz
                  br_if 1 (;@6;)
                end
                local.get 84
                local.get 14
                local.get 24
                local.get 9
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                f32.load
                local.get 1
                local.get 17
                local.get 11
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                local.get 24
                local.get 10
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                f32.load
                f32.mul
                f32.add
                local.set 84
              end
              local.get 26
              local.get 17
              i32.const 2
              i32.shl
              i32.add
              local.get 84
              f32.store
              local.get 19
              local.get 15
              i32.add
              local.set 19
              local.get 17
              i32.const 1
              i32.add
              local.tee 17
              local.get 4
              i32.ne
              br_if 0 (;@5;)
            end
            local.get 2
            local.get 31
            i32.const 1
            i32.add
            local.tee 13
            local.get 4
            i32.mul
            i32.const 2
            i32.shl
            i32.add
            local.set 26
            local.get 0
            local.get 13
            local.get 8
            i32.mul
            i32.const 2
            i32.shl
            i32.add
            local.set 14
            i32.const 0
            local.set 17
            local.get 1
            local.set 19
            loop ;; label = @5
              f32.const 0x0p+0 (;=0;)
              local.set 84
              i32.const 0
              local.set 24
              block ;; label = @6
                block ;; label = @7
                  local.get 16
                  i32.eqz
                  br_if 0 (;@7;)
                  local.get 19
                  local.set 13
                  local.get 44
                  local.set 66
                  loop ;; label = @8
                    local.get 84
                    local.get 66
                    f32.load
                    local.get 13
                    f32.load
                    f32.mul
                    f32.add
                    local.get 66
                    local.get 22
                    i32.add
                    f32.load
                    local.get 13
                    local.get 67
                    i32.add
                    f32.load
                    f32.mul
                    f32.add
                    local.set 84
                    local.get 13
                    local.get 64
                    i32.add
                    local.set 13
                    local.get 66
                    local.get 65
                    i32.add
                    local.set 66
                    local.get 68
                    local.get 24
                    i32.const 2
                    i32.add
                    local.tee 24
                    i32.ne
                    br_if 0 (;@8;)
                  end
                  local.get 21
                  i32.eqz
                  br_if 1 (;@6;)
                end
                local.get 84
                local.get 14
                local.get 24
                local.get 9
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                f32.load
                local.get 1
                local.get 17
                local.get 11
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                local.get 24
                local.get 10
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                f32.load
                f32.mul
                f32.add
                local.set 84
              end
              local.get 26
              local.get 17
              i32.const 2
              i32.shl
              i32.add
              local.get 84
              f32.store
              local.get 19
              local.get 15
              i32.add
              local.set 19
              local.get 17
              i32.const 1
              i32.add
              local.tee 17
              local.get 4
              i32.ne
              br_if 0 (;@5;)
            end
            local.get 2
            local.get 31
            i32.const 2
            i32.add
            local.tee 13
            local.get 4
            i32.mul
            i32.const 2
            i32.shl
            i32.add
            local.set 26
            local.get 0
            local.get 13
            local.get 8
            i32.mul
            i32.const 2
            i32.shl
            i32.add
            local.set 14
            i32.const 0
            local.set 17
            local.get 1
            local.set 19
            loop ;; label = @5
              f32.const 0x0p+0 (;=0;)
              local.set 84
              i32.const 0
              local.set 24
              block ;; label = @6
                block ;; label = @7
                  local.get 16
                  i32.eqz
                  br_if 0 (;@7;)
                  local.get 19
                  local.set 13
                  local.get 63
                  local.set 66
                  loop ;; label = @8
                    local.get 84
                    local.get 66
                    f32.load
                    local.get 13
                    f32.load
                    f32.mul
                    f32.add
                    local.get 66
                    local.get 22
                    i32.add
                    f32.load
                    local.get 13
                    local.get 67
                    i32.add
                    f32.load
                    f32.mul
                    f32.add
                    local.set 84
                    local.get 13
                    local.get 64
                    i32.add
                    local.set 13
                    local.get 66
                    local.get 65
                    i32.add
                    local.set 66
                    local.get 68
                    local.get 24
                    i32.const 2
                    i32.add
                    local.tee 24
                    i32.ne
                    br_if 0 (;@8;)
                  end
                  local.get 21
                  i32.eqz
                  br_if 1 (;@6;)
                end
                local.get 84
                local.get 14
                local.get 24
                local.get 9
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                f32.load
                local.get 1
                local.get 17
                local.get 11
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                local.get 24
                local.get 10
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                f32.load
                f32.mul
                f32.add
                local.set 84
              end
              local.get 26
              local.get 17
              i32.const 2
              i32.shl
              i32.add
              local.get 84
              f32.store
              local.get 19
              local.get 15
              i32.add
              local.set 19
              local.get 17
              i32.const 1
              i32.add
              local.tee 17
              local.get 4
              i32.ne
              br_if 0 (;@5;)
            end
            local.get 2
            local.get 31
            i32.const 3
            i32.add
            local.tee 13
            local.get 4
            i32.mul
            i32.const 2
            i32.shl
            i32.add
            local.set 26
            local.get 0
            local.get 13
            local.get 8
            i32.mul
            i32.const 2
            i32.shl
            i32.add
            local.set 14
            i32.const 0
            local.set 17
            local.get 1
            local.set 19
            loop ;; label = @5
              f32.const 0x0p+0 (;=0;)
              local.set 84
              i32.const 0
              local.set 24
              block ;; label = @6
                block ;; label = @7
                  local.get 16
                  i32.eqz
                  br_if 0 (;@7;)
                  local.get 19
                  local.set 13
                  local.get 12
                  local.set 66
                  loop ;; label = @8
                    local.get 84
                    local.get 66
                    f32.load
                    local.get 13
                    f32.load
                    f32.mul
                    f32.add
                    local.get 66
                    local.get 22
                    i32.add
                    f32.load
                    local.get 13
                    local.get 67
                    i32.add
                    f32.load
                    f32.mul
                    f32.add
                    local.set 84
                    local.get 13
                    local.get 64
                    i32.add
                    local.set 13
                    local.get 66
                    local.get 65
                    i32.add
                    local.set 66
                    local.get 68
                    local.get 24
                    i32.const 2
                    i32.add
                    local.tee 24
                    i32.ne
                    br_if 0 (;@8;)
                  end
                  local.get 21
                  i32.eqz
                  br_if 1 (;@6;)
                end
                local.get 84
                local.get 14
                local.get 24
                local.get 9
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                f32.load
                local.get 1
                local.get 17
                local.get 11
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                local.get 24
                local.get 10
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                f32.load
                f32.mul
                f32.add
                local.set 84
              end
              local.get 26
              local.get 17
              i32.const 2
              i32.shl
              i32.add
              local.get 84
              f32.store
              local.get 19
              local.get 15
              i32.add
              local.set 19
              local.get 17
              i32.const 1
              i32.add
              local.tee 17
              local.get 4
              i32.ne
              br_if 0 (;@5;)
            end
          end
          local.get 12
          local.get 33
          i32.add
          local.set 12
          local.get 63
          local.get 33
          i32.add
          local.set 63
          local.get 44
          local.get 33
          i32.add
          local.set 44
          local.get 59
          local.get 33
          i32.add
          local.set 59
          local.get 6
          local.set 31
          local.get 6
          i32.const 4
          i32.add
          local.tee 13
          local.get 7
          i32.le_u
          br_if 0 (;@3;)
          br 2 (;@1;)
        end
      end
      local.get 4
      i32.const 4
      i32.shl
      local.set 22
      local.get 2
      local.get 6
      local.get 4
      i32.mul
      i32.const 2
      i32.shl
      i32.add
      local.set 66
      local.get 6
      local.set 13
      loop ;; label = @2
        block ;; label = @3
          local.get 13
          i32.const -5
          i32.gt_u
          br_if 0 (;@3;)
          local.get 22
          i32.eqz
          br_if 0 (;@3;)
          local.get 66
          i32.const 0
          local.get 22
          memory.fill
        end
        local.get 66
        local.get 22
        i32.add
        local.set 66
        local.get 13
        i32.const 8
        i32.add
        local.set 67
        local.get 13
        i32.const 4
        i32.add
        local.tee 6
        local.set 13
        local.get 67
        local.get 7
        i32.le_u
        br_if 0 (;@2;)
      end
    end
    block ;; label = @1
      local.get 4
      i32.eqz
      br_if 0 (;@1;)
      local.get 7
      local.get 6
      i32.le_u
      br_if 0 (;@1;)
      block ;; label = @2
        local.get 5
        i32.eqz
        br_if 0 (;@2;)
        local.get 10
        i32.const 3
        i32.shl
        local.set 22
        local.get 11
        i32.const 2
        i32.shl
        local.set 19
        local.get 10
        i32.const 2
        i32.shl
        local.set 64
        local.get 9
        i32.const 3
        i32.shl
        local.set 65
        local.get 8
        i32.const 2
        i32.shl
        local.set 12
        local.get 9
        i32.const 2
        i32.shl
        local.set 68
        local.get 5
        i32.const -2
        i32.and
        local.set 24
        local.get 5
        i32.const 1
        i32.and
        local.set 59
        local.get 0
        local.get 6
        local.get 8
        i32.mul
        i32.const 2
        i32.shl
        i32.add
        local.set 26
        loop ;; label = @3
          local.get 2
          local.get 6
          local.get 4
          i32.mul
          i32.const 2
          i32.shl
          i32.add
          local.set 17
          local.get 0
          local.get 6
          local.get 8
          i32.mul
          i32.const 2
          i32.shl
          i32.add
          local.set 21
          i32.const 0
          local.set 15
          local.get 1
          local.set 16
          loop ;; label = @4
            f32.const 0x0p+0 (;=0;)
            local.set 84
            i32.const 0
            local.set 67
            block ;; label = @5
              block ;; label = @6
                local.get 5
                i32.const 1
                i32.eq
                br_if 0 (;@6;)
                f32.const 0x0p+0 (;=0;)
                local.set 84
                i32.const 0
                local.set 67
                local.get 16
                local.set 13
                local.get 26
                local.set 66
                loop ;; label = @7
                  local.get 84
                  local.get 66
                  f32.load
                  local.get 13
                  f32.load
                  f32.mul
                  f32.add
                  local.get 66
                  local.get 68
                  i32.add
                  f32.load
                  local.get 13
                  local.get 64
                  i32.add
                  f32.load
                  f32.mul
                  f32.add
                  local.set 84
                  local.get 13
                  local.get 22
                  i32.add
                  local.set 13
                  local.get 66
                  local.get 65
                  i32.add
                  local.set 66
                  local.get 24
                  local.get 67
                  i32.const 2
                  i32.add
                  local.tee 67
                  i32.ne
                  br_if 0 (;@7;)
                end
                local.get 59
                i32.eqz
                br_if 1 (;@5;)
              end
              local.get 84
              local.get 21
              local.get 67
              local.get 9
              i32.mul
              i32.const 2
              i32.shl
              i32.add
              f32.load
              local.get 1
              local.get 15
              local.get 11
              i32.mul
              i32.const 2
              i32.shl
              i32.add
              local.get 67
              local.get 10
              i32.mul
              i32.const 2
              i32.shl
              i32.add
              f32.load
              f32.mul
              f32.add
              local.set 84
            end
            local.get 17
            local.get 15
            i32.const 2
            i32.shl
            i32.add
            local.get 84
            f32.store
            local.get 16
            local.get 19
            i32.add
            local.set 16
            local.get 15
            i32.const 1
            i32.add
            local.tee 15
            local.get 4
            i32.ne
            br_if 0 (;@4;)
          end
          local.get 26
          local.get 12
          i32.add
          local.set 26
          local.get 6
          i32.const 1
          i32.add
          local.tee 6
          local.get 7
          i32.ne
          br_if 0 (;@3;)
          br 2 (;@1;)
        end
      end
      local.get 4
      local.get 7
      local.get 6
      i32.sub
      i32.mul
      i32.const 2
      i32.shl
      local.tee 13
      i32.eqz
      br_if 0 (;@1;)
      local.get 2
      local.get 4
      local.get 6
      i32.mul
      i32.const 2
      i32.shl
      i32.add
      i32.const 0
      local.get 13
      memory.fill
    end
  )
  (func (;7;) (type 2) (param i32 i32 i32 i32 i32 i32 i32 i32 i32)
    (local i32 i32)
    block ;; label = @1
      block ;; label = @2
        local.get 7
        local.get 8
        i32.ge_u
        br_if 0 (;@2;)
        local.get 6
        local.get 3
        i32.mul
        local.tee 9
        i32.eqz
        br_if 1 (;@1;)
        local.get 6
        i32.eqz
        br_if 1 (;@1;)
        local.get 6
        local.get 4
        i32.mul
        local.set 10
        local.get 0
        local.get 7
        i32.const 2
        i32.shl
        i32.add
        local.set 3
        loop ;; label = @3
          local.get 1
          local.get 10
          local.get 7
          local.get 9
          i32.div_u
          local.tee 4
          i32.mul
          i32.const 2
          i32.shl
          i32.add
          local.get 7
          local.get 4
          local.get 9
          i32.mul
          i32.sub
          local.tee 4
          local.get 6
          i32.div_u
          local.tee 0
          local.get 5
          i32.add
          local.get 6
          i32.mul
          i32.const 2
          i32.shl
          i32.add
          local.get 4
          local.get 0
          local.get 6
          i32.mul
          i32.sub
          i32.const 2
          i32.shl
          i32.add
          local.get 3
          f32.load
          f32.store
          local.get 3
          i32.const 4
          i32.add
          local.set 3
          local.get 8
          local.get 7
          i32.const 1
          i32.add
          local.tee 7
          i32.ne
          br_if 0 (;@3;)
        end
      end
      return
    end
    call 3
    unreachable
  )
  (func (;8;) (type 5) (param i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
    (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 f32 i32 i32 i32 i32 f32 i32 i32 v128)
    block ;; label = @1
      local.get 14
      i32.eqz
      br_if 0 (;@1;)
      local.get 8
      local.get 14
      i32.div_u
      local.set 17
      local.get 6
      local.get 14
      i32.div_u
      local.set 18
      block ;; label = @2
        local.get 16
        local.get 15
        i32.le_u
        br_if 0 (;@2;)
        local.get 8
        i32.eqz
        br_if 0 (;@2;)
        local.get 14
        local.get 8
        i32.gt_u
        br_if 1 (;@1;)
        local.get 10
        i32.eqz
        br_if 0 (;@2;)
        local.get 10
        local.get 8
        i32.mul
        local.set 19
        block ;; label = @3
          block ;; label = @4
            local.get 14
            local.get 6
            i32.gt_u
            br_if 0 (;@4;)
            block ;; label = @5
              local.get 9
              i32.eqz
              br_if 0 (;@5;)
              local.get 7
              local.get 6
              i32.mul
              local.set 20
              local.get 18
              local.get 9
              i32.mul
              local.set 21
              local.get 9
              i32.const -2
              i32.and
              local.set 22
              local.get 9
              i32.const 1
              i32.and
              local.set 23
              block ;; label = @6
                local.get 4
                i32.eqz
                br_if 0 (;@6;)
                local.get 13
                i32.const 1
                i32.shl
                local.set 24
                i32.const 0
                local.get 12
                i32.sub
                local.set 25
                loop ;; label = @7
                  local.get 3
                  local.get 19
                  local.get 15
                  i32.mul
                  i32.const 2
                  i32.shl
                  i32.add
                  local.set 26
                  local.get 0
                  local.get 20
                  local.get 15
                  i32.mul
                  i32.const 2
                  i32.shl
                  i32.add
                  local.set 27
                  i32.const 0
                  local.set 28
                  loop ;; label = @8
                    local.get 28
                    local.get 17
                    i32.div_u
                    local.get 18
                    i32.mul
                    local.set 29
                    local.get 1
                    local.get 21
                    local.get 28
                    i32.mul
                    i32.const 2
                    i32.shl
                    i32.add
                    local.set 30
                    local.get 26
                    local.get 28
                    local.get 10
                    i32.mul
                    i32.const 2
                    i32.shl
                    i32.add
                    local.set 31
                    local.get 2
                    local.get 28
                    i32.const 2
                    i32.shl
                    i32.add
                    f32.load
                    local.set 32
                    i32.const 0
                    local.set 33
                    local.get 25
                    local.set 34
                    loop ;; label = @9
                      local.get 33
                      local.get 11
                      i32.mul
                      local.get 12
                      i32.sub
                      local.set 35
                      i32.const 0
                      local.set 36
                      local.get 32
                      local.set 37
                      loop ;; label = @10
                        local.get 30
                        local.get 36
                        local.get 9
                        i32.mul
                        i32.const 2
                        i32.shl
                        i32.add
                        local.set 38
                        local.get 27
                        local.get 36
                        local.get 29
                        i32.add
                        local.get 7
                        i32.mul
                        i32.const 2
                        i32.shl
                        i32.add
                        local.set 39
                        i32.const 0
                        local.set 6
                        block ;; label = @11
                          block ;; label = @12
                            local.get 9
                            i32.const 1
                            i32.eq
                            br_if 0 (;@12;)
                            i32.const 0
                            local.set 6
                            local.get 34
                            local.set 14
                            loop ;; label = @13
                              block ;; label = @14
                                local.get 14
                                i32.const 0
                                i32.lt_s
                                br_if 0 (;@14;)
                                local.get 14
                                local.get 7
                                i32.ge_s
                                br_if 0 (;@14;)
                                local.get 37
                                local.get 39
                                local.get 35
                                local.get 6
                                local.get 13
                                i32.mul
                                i32.add
                                i32.const 2
                                i32.shl
                                i32.add
                                f32.load
                                local.get 38
                                local.get 6
                                i32.const 2
                                i32.shl
                                i32.add
                                f32.load
                                f32.mul
                                f32.add
                                local.set 37
                              end
                              block ;; label = @14
                                local.get 13
                                local.get 14
                                i32.add
                                local.tee 4
                                i32.const 0
                                i32.lt_s
                                br_if 0 (;@14;)
                                local.get 4
                                local.get 7
                                i32.ge_s
                                br_if 0 (;@14;)
                                local.get 37
                                local.get 39
                                local.get 35
                                local.get 6
                                i32.const 1
                                i32.or
                                local.tee 4
                                local.get 13
                                i32.mul
                                i32.add
                                i32.const 2
                                i32.shl
                                i32.add
                                f32.load
                                local.get 38
                                local.get 4
                                i32.const 2
                                i32.shl
                                i32.add
                                f32.load
                                f32.mul
                                f32.add
                                local.set 37
                              end
                              local.get 14
                              local.get 24
                              i32.add
                              local.set 14
                              local.get 22
                              local.get 6
                              i32.const 2
                              i32.add
                              local.tee 6
                              i32.ne
                              br_if 0 (;@13;)
                            end
                            local.get 23
                            i32.eqz
                            br_if 1 (;@11;)
                          end
                          local.get 35
                          local.get 6
                          local.get 13
                          i32.mul
                          i32.add
                          local.tee 14
                          i32.const 0
                          i32.lt_s
                          br_if 0 (;@11;)
                          local.get 14
                          local.get 7
                          i32.ge_s
                          br_if 0 (;@11;)
                          local.get 37
                          local.get 39
                          local.get 14
                          i32.const 2
                          i32.shl
                          i32.add
                          f32.load
                          local.get 38
                          local.get 6
                          i32.const 2
                          i32.shl
                          i32.add
                          f32.load
                          f32.mul
                          f32.add
                          local.set 37
                        end
                        local.get 36
                        i32.const 1
                        i32.add
                        local.tee 36
                        local.get 18
                        i32.lt_u
                        br_if 0 (;@10;)
                      end
                      local.get 31
                      local.get 33
                      i32.const 2
                      i32.shl
                      i32.add
                      local.get 37
                      f32.store
                      local.get 34
                      local.get 11
                      i32.add
                      local.set 34
                      local.get 33
                      i32.const 1
                      i32.add
                      local.tee 33
                      local.get 10
                      i32.ne
                      br_if 0 (;@9;)
                    end
                    local.get 28
                    i32.const 1
                    i32.add
                    local.tee 28
                    local.get 8
                    i32.ne
                    br_if 0 (;@8;)
                  end
                  local.get 15
                  i32.const 1
                  i32.add
                  local.tee 15
                  local.get 16
                  i32.ne
                  br_if 0 (;@7;)
                  br 5 (;@2;)
                end
              end
              local.get 13
              i32.const 1
              i32.shl
              local.set 24
              i32.const 0
              local.get 12
              i32.sub
              local.set 26
              loop ;; label = @6
                local.get 3
                local.get 19
                local.get 15
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                local.set 2
                local.get 0
                local.get 20
                local.get 15
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                local.set 27
                i32.const 0
                local.set 28
                loop ;; label = @7
                  local.get 28
                  local.get 17
                  i32.div_u
                  local.get 18
                  i32.mul
                  local.set 29
                  local.get 1
                  local.get 21
                  local.get 28
                  i32.mul
                  i32.const 2
                  i32.shl
                  i32.add
                  local.set 30
                  local.get 2
                  local.get 28
                  local.get 10
                  i32.mul
                  i32.const 2
                  i32.shl
                  i32.add
                  local.set 31
                  i32.const 0
                  local.set 33
                  local.get 26
                  local.set 34
                  loop ;; label = @8
                    local.get 33
                    local.get 11
                    i32.mul
                    local.get 12
                    i32.sub
                    local.set 35
                    f32.const 0x0p+0 (;=0;)
                    local.set 37
                    i32.const 0
                    local.set 36
                    loop ;; label = @9
                      local.get 30
                      local.get 36
                      local.get 9
                      i32.mul
                      i32.const 2
                      i32.shl
                      i32.add
                      local.set 38
                      local.get 27
                      local.get 36
                      local.get 29
                      i32.add
                      local.get 7
                      i32.mul
                      i32.const 2
                      i32.shl
                      i32.add
                      local.set 39
                      i32.const 0
                      local.set 6
                      block ;; label = @10
                        block ;; label = @11
                          local.get 9
                          i32.const 1
                          i32.eq
                          br_if 0 (;@11;)
                          i32.const 0
                          local.set 6
                          local.get 34
                          local.set 14
                          loop ;; label = @12
                            block ;; label = @13
                              local.get 14
                              i32.const 0
                              i32.lt_s
                              br_if 0 (;@13;)
                              local.get 14
                              local.get 7
                              i32.ge_s
                              br_if 0 (;@13;)
                              local.get 37
                              local.get 39
                              local.get 35
                              local.get 6
                              local.get 13
                              i32.mul
                              i32.add
                              i32.const 2
                              i32.shl
                              i32.add
                              f32.load
                              local.get 38
                              local.get 6
                              i32.const 2
                              i32.shl
                              i32.add
                              f32.load
                              f32.mul
                              f32.add
                              local.set 37
                            end
                            block ;; label = @13
                              local.get 13
                              local.get 14
                              i32.add
                              local.tee 4
                              i32.const 0
                              i32.lt_s
                              br_if 0 (;@13;)
                              local.get 4
                              local.get 7
                              i32.ge_s
                              br_if 0 (;@13;)
                              local.get 37
                              local.get 39
                              local.get 35
                              local.get 6
                              i32.const 1
                              i32.or
                              local.tee 4
                              local.get 13
                              i32.mul
                              i32.add
                              i32.const 2
                              i32.shl
                              i32.add
                              f32.load
                              local.get 38
                              local.get 4
                              i32.const 2
                              i32.shl
                              i32.add
                              f32.load
                              f32.mul
                              f32.add
                              local.set 37
                            end
                            local.get 14
                            local.get 24
                            i32.add
                            local.set 14
                            local.get 22
                            local.get 6
                            i32.const 2
                            i32.add
                            local.tee 6
                            i32.ne
                            br_if 0 (;@12;)
                          end
                          local.get 23
                          i32.eqz
                          br_if 1 (;@10;)
                        end
                        local.get 35
                        local.get 6
                        local.get 13
                        i32.mul
                        i32.add
                        local.tee 14
                        i32.const 0
                        i32.lt_s
                        br_if 0 (;@10;)
                        local.get 14
                        local.get 7
                        i32.ge_s
                        br_if 0 (;@10;)
                        local.get 37
                        local.get 39
                        local.get 14
                        i32.const 2
                        i32.shl
                        i32.add
                        f32.load
                        local.get 38
                        local.get 6
                        i32.const 2
                        i32.shl
                        i32.add
                        f32.load
                        f32.mul
                        f32.add
                        local.set 37
                      end
                      local.get 36
                      i32.const 1
                      i32.add
                      local.tee 36
                      local.get 18
                      i32.lt_u
                      br_if 0 (;@9;)
                    end
                    local.get 31
                    local.get 33
                    i32.const 2
                    i32.shl
                    i32.add
                    local.get 37
                    f32.store
                    local.get 34
                    local.get 11
                    i32.add
                    local.set 34
                    local.get 33
                    i32.const 1
                    i32.add
                    local.tee 33
                    local.get 10
                    i32.ne
                    br_if 0 (;@8;)
                  end
                  local.get 28
                  i32.const 1
                  i32.add
                  local.tee 28
                  local.get 8
                  i32.ne
                  br_if 0 (;@7;)
                end
                local.get 15
                i32.const 1
                i32.add
                local.tee 15
                local.get 16
                i32.ne
                br_if 0 (;@6;)
                br 4 (;@2;)
              end
            end
            local.get 4
            i32.eqz
            br_if 1 (;@3;)
            local.get 10
            i32.const 2
            i32.shl
            local.set 22
            local.get 10
            i32.const -4
            i32.and
            local.set 4
            local.get 19
            i32.const 2
            i32.shl
            local.set 35
            local.get 3
            local.get 19
            local.get 15
            i32.mul
            i32.const 2
            i32.shl
            i32.add
            local.set 39
            local.get 10
            i32.const 4
            i32.lt_u
            local.set 38
            loop ;; label = @5
              i32.const 0
              local.set 7
              local.get 39
              local.set 13
              loop ;; label = @6
                local.get 2
                local.get 7
                i32.const 2
                i32.shl
                i32.add
                f32.load
                local.set 37
                i32.const 0
                local.set 14
                block ;; label = @7
                  block ;; label = @8
                    local.get 38
                    br_if 0 (;@8;)
                    local.get 37
                    f32x4.splat
                    local.set 40
                    local.get 13
                    local.set 14
                    local.get 4
                    local.set 6
                    loop ;; label = @9
                      local.get 14
                      local.get 40
                      v128.store align=4
                      local.get 14
                      i32.const 16
                      i32.add
                      local.set 14
                      local.get 6
                      i32.const -4
                      i32.add
                      local.tee 6
                      br_if 0 (;@9;)
                    end
                    local.get 4
                    local.set 14
                    local.get 10
                    local.get 4
                    i32.eq
                    br_if 1 (;@7;)
                  end
                  local.get 10
                  local.get 14
                  i32.sub
                  local.set 6
                  local.get 13
                  local.get 14
                  i32.const 2
                  i32.shl
                  i32.add
                  local.set 14
                  loop ;; label = @8
                    local.get 14
                    local.get 37
                    f32.store
                    local.get 14
                    i32.const 4
                    i32.add
                    local.set 14
                    local.get 6
                    i32.const -1
                    i32.add
                    local.tee 6
                    br_if 0 (;@8;)
                  end
                end
                local.get 13
                local.get 22
                i32.add
                local.set 13
                local.get 7
                i32.const 1
                i32.add
                local.tee 7
                local.get 8
                i32.ne
                br_if 0 (;@6;)
              end
              local.get 39
              local.get 35
              i32.add
              local.set 39
              local.get 15
              i32.const 1
              i32.add
              local.tee 15
              local.get 16
              i32.ne
              br_if 0 (;@5;)
              br 3 (;@2;)
            end
          end
          local.get 4
          i32.eqz
          br_if 0 (;@3;)
          local.get 10
          i32.const 2
          i32.shl
          local.set 22
          local.get 10
          i32.const -4
          i32.and
          local.set 4
          local.get 19
          i32.const 2
          i32.shl
          local.set 35
          local.get 3
          local.get 19
          local.get 15
          i32.mul
          i32.const 2
          i32.shl
          i32.add
          local.set 39
          local.get 10
          i32.const 4
          i32.lt_u
          local.set 38
          loop ;; label = @4
            i32.const 0
            local.set 7
            local.get 39
            local.set 13
            loop ;; label = @5
              local.get 2
              local.get 7
              i32.const 2
              i32.shl
              i32.add
              f32.load
              local.set 37
              i32.const 0
              local.set 14
              block ;; label = @6
                block ;; label = @7
                  local.get 38
                  br_if 0 (;@7;)
                  local.get 37
                  f32x4.splat
                  local.set 40
                  local.get 13
                  local.set 14
                  local.get 4
                  local.set 6
                  loop ;; label = @8
                    local.get 14
                    local.get 40
                    v128.store align=4
                    local.get 14
                    i32.const 16
                    i32.add
                    local.set 14
                    local.get 6
                    i32.const -4
                    i32.add
                    local.tee 6
                    br_if 0 (;@8;)
                  end
                  local.get 4
                  local.set 14
                  local.get 10
                  local.get 4
                  i32.eq
                  br_if 1 (;@6;)
                end
                local.get 10
                local.get 14
                i32.sub
                local.set 6
                local.get 13
                local.get 14
                i32.const 2
                i32.shl
                i32.add
                local.set 14
                loop ;; label = @7
                  local.get 14
                  local.get 37
                  f32.store
                  local.get 14
                  i32.const 4
                  i32.add
                  local.set 14
                  local.get 6
                  i32.const -1
                  i32.add
                  local.tee 6
                  br_if 0 (;@7;)
                end
              end
              local.get 13
              local.get 22
              i32.add
              local.set 13
              local.get 7
              i32.const 1
              i32.add
              local.tee 7
              local.get 8
              i32.ne
              br_if 0 (;@5;)
            end
            local.get 39
            local.get 35
            i32.add
            local.set 39
            local.get 15
            i32.const 1
            i32.add
            local.tee 15
            local.get 16
            i32.ne
            br_if 0 (;@4;)
            br 2 (;@2;)
          end
        end
        local.get 19
        local.get 16
        local.get 15
        i32.sub
        i32.mul
        i32.const 2
        i32.shl
        local.tee 14
        i32.eqz
        br_if 0 (;@2;)
        local.get 3
        local.get 15
        local.get 10
        i32.mul
        local.get 8
        i32.mul
        i32.const 2
        i32.shl
        i32.add
        i32.const 0
        local.get 14
        memory.fill
      end
      return
    end
    call 3
    unreachable
  )
  (func (;9;) (type 5) (param i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
    (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 f32 v128 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
    block ;; label = @1
      local.get 14
      i32.eqz
      br_if 0 (;@1;)
      local.get 6
      local.get 14
      i32.div_u
      local.set 17
      local.get 8
      local.get 14
      i32.div_u
      local.set 18
      block ;; label = @2
        block ;; label = @3
          local.get 16
          local.get 15
          i32.le_u
          br_if 0 (;@3;)
          local.get 10
          local.get 8
          i32.mul
          local.set 19
          block ;; label = @4
            local.get 6
            i32.eqz
            br_if 0 (;@4;)
            local.get 14
            local.get 6
            i32.gt_u
            br_if 2 (;@2;)
            local.get 7
            local.get 6
            i32.mul
            local.set 20
            local.get 18
            local.get 9
            i32.mul
            local.set 21
            i32.const 0
            local.get 12
            i32.sub
            local.set 22
            local.get 10
            i32.const 2
            i32.shl
            local.set 23
            local.get 10
            i32.const -4
            i32.and
            local.set 24
            local.get 16
            local.get 15
            i32.sub
            local.set 25
            local.get 19
            i32.const 2
            i32.shl
            local.set 26
            local.get 14
            local.get 8
            i32.gt_u
            local.get 9
            i32.eqz
            i32.or
            local.set 27
            local.get 8
            i32.eqz
            local.get 10
            i32.eqz
            i32.or
            local.set 28
            local.get 10
            i32.const 4
            i32.lt_u
            local.set 29
            local.get 3
            local.get 19
            local.get 15
            i32.mul
            i32.const 2
            i32.shl
            i32.add
            local.tee 30
            local.set 31
            i32.const 0
            local.set 32
            loop ;; label = @5
              block ;; label = @6
                local.get 28
                br_if 0 (;@6;)
                block ;; label = @7
                  local.get 4
                  i32.eqz
                  br_if 0 (;@7;)
                  i32.const 0
                  local.set 33
                  local.get 31
                  local.set 34
                  loop ;; label = @8
                    local.get 2
                    local.get 33
                    i32.const 2
                    i32.shl
                    i32.add
                    f32.load
                    local.set 35
                    i32.const 0
                    local.set 14
                    block ;; label = @9
                      block ;; label = @10
                        local.get 29
                        br_if 0 (;@10;)
                        local.get 35
                        f32x4.splat
                        local.set 36
                        local.get 34
                        local.set 14
                        local.get 24
                        local.set 37
                        loop ;; label = @11
                          local.get 14
                          local.get 36
                          v128.store align=4
                          local.get 14
                          i32.const 16
                          i32.add
                          local.set 14
                          local.get 37
                          i32.const -4
                          i32.add
                          local.tee 37
                          br_if 0 (;@11;)
                        end
                        local.get 24
                        local.set 14
                        local.get 10
                        local.get 24
                        i32.eq
                        br_if 1 (;@9;)
                      end
                      local.get 10
                      local.get 14
                      i32.sub
                      local.set 37
                      local.get 34
                      local.get 14
                      i32.const 2
                      i32.shl
                      i32.add
                      local.set 14
                      loop ;; label = @10
                        local.get 14
                        local.get 35
                        f32.store
                        local.get 14
                        i32.const 4
                        i32.add
                        local.set 14
                        local.get 37
                        i32.const -1
                        i32.add
                        local.tee 37
                        br_if 0 (;@10;)
                      end
                    end
                    local.get 34
                    local.get 23
                    i32.add
                    local.set 34
                    local.get 33
                    i32.const 1
                    i32.add
                    local.tee 33
                    local.get 8
                    i32.eq
                    br_if 2 (;@6;)
                    br 0 (;@8;)
                  end
                end
                local.get 26
                i32.eqz
                br_if 0 (;@6;)
                local.get 30
                local.get 26
                local.get 32
                i32.mul
                i32.add
                i32.const 0
                local.get 26
                memory.fill
              end
              block ;; label = @6
                local.get 7
                i32.eqz
                br_if 0 (;@6;)
                local.get 3
                local.get 19
                local.get 15
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                local.set 38
                local.get 0
                local.get 20
                local.get 15
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                local.set 39
                i32.const 0
                local.set 40
                loop ;; label = @7
                  local.get 40
                  local.get 17
                  i32.div_u
                  local.set 14
                  block ;; label = @8
                    local.get 27
                    br_if 0 (;@8;)
                    local.get 14
                    local.get 18
                    i32.mul
                    local.set 41
                    local.get 1
                    local.get 21
                    local.get 40
                    i32.mul
                    i32.const 2
                    i32.shl
                    i32.add
                    local.set 42
                    local.get 39
                    local.get 40
                    local.get 7
                    i32.mul
                    i32.const 2
                    i32.shl
                    i32.add
                    local.set 43
                    i32.const 0
                    local.set 44
                    local.get 22
                    local.set 45
                    loop ;; label = @9
                      block ;; label = @10
                        local.get 43
                        local.get 44
                        i32.const 2
                        i32.shl
                        i32.add
                        f32.load
                        local.tee 35
                        f32.const 0x0p+0 (;=0;)
                        f32.eq
                        br_if 0 (;@10;)
                        local.get 44
                        local.get 11
                        i32.mul
                        local.get 12
                        i32.sub
                        local.set 46
                        i32.const 0
                        local.set 16
                        loop ;; label = @11
                          local.get 42
                          local.get 16
                          local.get 9
                          i32.mul
                          i32.const 2
                          i32.shl
                          i32.add
                          local.set 47
                          local.get 38
                          local.get 16
                          local.get 41
                          i32.add
                          local.get 10
                          i32.mul
                          i32.const 2
                          i32.shl
                          i32.add
                          local.set 48
                          local.get 9
                          local.set 34
                          local.get 45
                          local.set 14
                          i32.const 0
                          local.set 37
                          loop ;; label = @12
                            block ;; label = @13
                              local.get 14
                              i32.const 0
                              i32.lt_s
                              br_if 0 (;@13;)
                              local.get 14
                              local.get 10
                              i32.ge_s
                              br_if 0 (;@13;)
                              local.get 48
                              local.get 46
                              local.get 37
                              local.get 13
                              i32.mul
                              i32.add
                              i32.const 2
                              i32.shl
                              i32.add
                              local.tee 33
                              local.get 33
                              f32.load
                              local.get 35
                              local.get 47
                              local.get 37
                              i32.const 2
                              i32.shl
                              i32.add
                              f32.load
                              f32.mul
                              f32.add
                              f32.store
                            end
                            local.get 37
                            i32.const 1
                            i32.add
                            local.set 37
                            local.get 14
                            local.get 13
                            i32.add
                            local.set 14
                            local.get 34
                            i32.const -1
                            i32.add
                            local.tee 34
                            br_if 0 (;@12;)
                          end
                          local.get 16
                          i32.const 1
                          i32.add
                          local.tee 16
                          local.get 18
                          i32.lt_u
                          br_if 0 (;@11;)
                        end
                      end
                      local.get 45
                      local.get 11
                      i32.add
                      local.set 45
                      local.get 44
                      i32.const 1
                      i32.add
                      local.tee 44
                      local.get 7
                      i32.ne
                      br_if 0 (;@9;)
                    end
                  end
                  local.get 40
                  i32.const 1
                  i32.add
                  local.tee 40
                  local.get 6
                  i32.ne
                  br_if 0 (;@7;)
                end
              end
              local.get 15
              i32.const 1
              i32.add
              local.set 15
              local.get 31
              local.get 26
              i32.add
              local.set 31
              local.get 32
              i32.const 1
              i32.add
              local.tee 32
              local.get 25
              i32.ne
              br_if 0 (;@5;)
              br 2 (;@3;)
            end
          end
          local.get 8
          i32.eqz
          br_if 0 (;@3;)
          local.get 10
          i32.eqz
          br_if 0 (;@3;)
          block ;; label = @4
            local.get 4
            i32.eqz
            br_if 0 (;@4;)
            local.get 10
            i32.const 2
            i32.shl
            local.set 47
            local.get 10
            i32.const -4
            i32.and
            local.set 33
            local.get 19
            i32.const 2
            i32.shl
            local.set 9
            local.get 3
            local.get 19
            local.get 15
            i32.mul
            i32.const 2
            i32.shl
            i32.add
            local.set 46
            local.get 10
            i32.const 4
            i32.lt_u
            local.set 48
            loop ;; label = @5
              i32.const 0
              local.set 13
              local.get 46
              local.set 34
              loop ;; label = @6
                local.get 2
                local.get 13
                i32.const 2
                i32.shl
                i32.add
                f32.load
                local.set 35
                i32.const 0
                local.set 14
                block ;; label = @7
                  block ;; label = @8
                    local.get 48
                    br_if 0 (;@8;)
                    local.get 35
                    f32x4.splat
                    local.set 36
                    local.get 34
                    local.set 14
                    local.get 33
                    local.set 37
                    loop ;; label = @9
                      local.get 14
                      local.get 36
                      v128.store align=4
                      local.get 14
                      i32.const 16
                      i32.add
                      local.set 14
                      local.get 37
                      i32.const -4
                      i32.add
                      local.tee 37
                      br_if 0 (;@9;)
                    end
                    local.get 33
                    local.set 14
                    local.get 10
                    local.get 33
                    i32.eq
                    br_if 1 (;@7;)
                  end
                  local.get 10
                  local.get 14
                  i32.sub
                  local.set 37
                  local.get 34
                  local.get 14
                  i32.const 2
                  i32.shl
                  i32.add
                  local.set 14
                  loop ;; label = @8
                    local.get 14
                    local.get 35
                    f32.store
                    local.get 14
                    i32.const 4
                    i32.add
                    local.set 14
                    local.get 37
                    i32.const -1
                    i32.add
                    local.tee 37
                    br_if 0 (;@8;)
                  end
                end
                local.get 34
                local.get 47
                i32.add
                local.set 34
                local.get 13
                i32.const 1
                i32.add
                local.tee 13
                local.get 8
                i32.ne
                br_if 0 (;@6;)
              end
              local.get 46
              local.get 9
              i32.add
              local.set 46
              local.get 15
              i32.const 1
              i32.add
              local.tee 15
              local.get 16
              i32.ne
              br_if 0 (;@5;)
              br 2 (;@3;)
            end
          end
          local.get 19
          local.get 16
          local.get 15
          i32.sub
          i32.mul
          i32.const 2
          i32.shl
          local.tee 14
          i32.eqz
          br_if 0 (;@3;)
          local.get 3
          local.get 15
          local.get 10
          i32.mul
          local.get 8
          i32.mul
          i32.const 2
          i32.shl
          i32.add
          i32.const 0
          local.get 14
          memory.fill
        end
        return
      end
      local.get 8
      i32.eqz
      br_if 0 (;@1;)
      local.get 10
      i32.eqz
      br_if 0 (;@1;)
      local.get 3
      local.get 19
      local.get 15
      i32.mul
      i32.const 2
      i32.shl
      i32.add
      local.set 34
      block ;; label = @2
        local.get 4
        i32.eqz
        br_if 0 (;@2;)
        local.get 10
        i32.const 2
        i32.shl
        local.set 47
        local.get 10
        i32.const -4
        i32.and
        local.set 33
        i32.const 0
        local.set 13
        local.get 10
        i32.const 4
        i32.lt_u
        local.set 48
        loop ;; label = @3
          local.get 2
          local.get 13
          i32.const 2
          i32.shl
          i32.add
          f32.load
          local.set 35
          i32.const 0
          local.set 14
          block ;; label = @4
            block ;; label = @5
              local.get 48
              br_if 0 (;@5;)
              local.get 35
              f32x4.splat
              local.set 36
              local.get 34
              local.set 14
              local.get 33
              local.set 37
              loop ;; label = @6
                local.get 14
                local.get 36
                v128.store align=4
                local.get 14
                i32.const 16
                i32.add
                local.set 14
                local.get 37
                i32.const -4
                i32.add
                local.tee 37
                br_if 0 (;@6;)
              end
              local.get 33
              local.set 14
              local.get 10
              local.get 33
              i32.eq
              br_if 1 (;@4;)
            end
            local.get 10
            local.get 14
            i32.sub
            local.set 37
            local.get 34
            local.get 14
            i32.const 2
            i32.shl
            i32.add
            local.set 14
            loop ;; label = @5
              local.get 14
              local.get 35
              f32.store
              local.get 14
              i32.const 4
              i32.add
              local.set 14
              local.get 37
              i32.const -1
              i32.add
              local.tee 37
              br_if 0 (;@5;)
            end
          end
          local.get 34
          local.get 47
          i32.add
          local.set 34
          local.get 13
          i32.const 1
          i32.add
          local.tee 13
          local.get 8
          i32.ne
          br_if 0 (;@3;)
          br 2 (;@1;)
        end
      end
      local.get 19
      i32.const 2
      i32.shl
      local.tee 14
      i32.eqz
      br_if 0 (;@1;)
      local.get 34
      i32.const 0
      local.get 14
      memory.fill
    end
    call 3
    unreachable
  )
  (func (;10;) (type 6) (param i32 i32 i32 i32)
    (local i32 i32 i32 i32 i32)
    block ;; label = @1
      local.get 2
      i32.const 4
      i32.add
      local.get 3
      i32.gt_u
      br_if 0 (;@1;)
      local.get 1
      local.get 2
      i32.const 2
      i32.shl
      local.tee 4
      i32.add
      local.set 5
      local.get 0
      local.get 4
      i32.add
      local.set 4
      loop ;; label = @2
        local.get 5
        local.get 4
        v128.load align=1
        v128.store align=1
        local.get 5
        i32.const 16
        i32.add
        local.set 5
        local.get 4
        i32.const 16
        i32.add
        local.set 4
        local.get 2
        local.tee 6
        i32.const 4
        i32.add
        local.set 2
        local.get 6
        i32.const 8
        i32.add
        local.get 3
        i32.le_u
        br_if 0 (;@2;)
      end
    end
    block ;; label = @1
      local.get 3
      local.get 2
      i32.le_u
      br_if 0 (;@1;)
      block ;; label = @2
        local.get 3
        local.get 2
        i32.sub
        local.tee 7
        i32.const 8
        i32.lt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 1
        i32.sub
        i32.const -16
        i32.gt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 2
        i32.const 2
        i32.shl
        local.tee 4
        i32.add
        local.set 5
        local.get 1
        local.get 4
        i32.add
        local.set 4
        local.get 2
        local.get 7
        i32.const -4
        i32.and
        local.tee 8
        i32.add
        local.set 2
        local.get 8
        local.set 6
        loop ;; label = @3
          local.get 4
          local.get 5
          v128.load align=4
          v128.store align=4
          local.get 5
          i32.const 16
          i32.add
          local.set 5
          local.get 4
          i32.const 16
          i32.add
          local.set 4
          local.get 6
          i32.const -4
          i32.add
          local.tee 6
          br_if 0 (;@3;)
        end
        local.get 7
        local.get 8
        i32.eq
        br_if 1 (;@1;)
      end
      local.get 2
      local.set 7
      block ;; label = @2
        local.get 3
        local.get 2
        i32.sub
        i32.const 3
        i32.and
        local.tee 6
        i32.eqz
        br_if 0 (;@2;)
        local.get 2
        local.get 6
        i32.add
        local.set 7
        local.get 0
        local.get 2
        i32.const 2
        i32.shl
        local.tee 4
        i32.add
        local.set 5
        local.get 1
        local.get 4
        i32.add
        local.set 4
        loop ;; label = @3
          local.get 4
          local.get 5
          f32.load
          f32.store
          local.get 5
          i32.const 4
          i32.add
          local.set 5
          local.get 4
          i32.const 4
          i32.add
          local.set 4
          local.get 6
          i32.const -1
          i32.add
          local.tee 6
          br_if 0 (;@3;)
        end
      end
      local.get 2
      local.get 3
      i32.sub
      i32.const -4
      i32.gt_u
      br_if 0 (;@1;)
      local.get 7
      i32.const 2
      i32.shl
      local.set 2
      local.get 3
      local.get 7
      i32.sub
      local.set 6
      loop ;; label = @2
        local.get 1
        local.get 2
        i32.add
        local.tee 5
        local.get 0
        local.get 2
        i32.add
        local.tee 4
        f32.load
        f32.store
        local.get 5
        i32.const 4
        i32.add
        local.get 4
        i32.const 4
        i32.add
        f32.load
        f32.store
        local.get 5
        i32.const 8
        i32.add
        local.get 4
        i32.const 8
        i32.add
        f32.load
        f32.store
        local.get 5
        i32.const 12
        i32.add
        local.get 4
        i32.const 12
        i32.add
        f32.load
        f32.store
        local.get 0
        i32.const 16
        i32.add
        local.set 0
        local.get 1
        i32.const 16
        i32.add
        local.set 1
        local.get 6
        i32.const -4
        i32.add
        local.tee 6
        br_if 0 (;@2;)
      end
    end
  )
  (func (;11;) (type 6) (param i32 i32 i32 i32)
    (local i32)
    block ;; label = @1
      local.get 3
      local.get 2
      i32.le_u
      br_if 0 (;@1;)
      local.get 3
      local.get 2
      i32.sub
      local.set 4
      local.get 0
      local.get 2
      i32.const 2
      i32.shl
      local.tee 3
      i32.add
      local.set 2
      local.get 1
      local.get 3
      i32.add
      local.set 3
      loop ;; label = @2
        local.get 3
        local.get 2
        f32.load
        call 12
        f32.store
        local.get 2
        i32.const 4
        i32.add
        local.set 2
        local.get 3
        i32.const 4
        i32.add
        local.set 3
        local.get 4
        i32.const -1
        i32.add
        local.tee 4
        br_if 0 (;@2;)
      end
    end
  )
  (func (;12;) (type 7) (param f32) (result f32)
    (local i32 f64 i32 i32 f64 i32 f64)
    global.get 0
    i32.const 16
    i32.sub
    local.tee 1
    global.set 0
    local.get 0
    f64.promote_f32
    local.set 2
    block ;; label = @1
      block ;; label = @2
        block ;; label = @3
          block ;; label = @4
            local.get 0
            i32.reinterpret_f32
            local.tee 3
            i32.const 2147483647
            i32.and
            local.tee 4
            i32.const 1061752795
            i32.lt_u
            br_if 0 (;@4;)
            block ;; label = @5
              local.get 4
              i32.const 1081824210
              i32.lt_u
              br_if 0 (;@5;)
              block ;; label = @6
                local.get 4
                i32.const 1088565718
                i32.lt_u
                br_if 0 (;@6;)
                block ;; label = @7
                  block ;; label = @8
                    block ;; label = @9
                      block ;; label = @10
                        block ;; label = @11
                          local.get 4
                          i32.const 2139095039
                          i32.gt_u
                          br_if 0 (;@11;)
                          local.get 1
                          i64.const 0
                          i64.store offset=8
                          block ;; label = @12
                            block ;; label = @13
                              local.get 4
                              i32.const 1305022426
                              i32.gt_u
                              br_if 0 (;@13;)
                              local.get 2
                              local.get 2
                              f64.const 0x1.45f306dc9c883p-1 (;=0.6366197723675814;)
                              f64.mul
                              f64.const 0x1.8p+52 (;=6755399441055744;)
                              f64.add
                              f64.const -0x1.8p+52 (;=-6755399441055744;)
                              f64.add
                              local.tee 5
                              f64.const -0x1.921fb5p+0 (;=-1.5707963109016418;)
                              f64.mul
                              f64.add
                              local.get 5
                              f64.const -0x1.110b4611a6263p-26 (;=-0.000000015893254773528196;)
                              f64.mul
                              f64.add
                              local.set 2
                              local.get 5
                              i32.trunc_sat_f64_s
                              local.set 4
                              br 1 (;@12;)
                            end
                            local.get 4
                            local.get 4
                            i32.const 23
                            i32.shr_u
                            i32.const -150
                            i32.add
                            local.tee 6
                            i32.const 23
                            i32.shl
                            i32.sub
                            f32.reinterpret_i32
                            f64.promote_f32
                            local.get 1
                            i32.const 8
                            i32.add
                            local.get 6
                            call 41
                            local.set 4
                            block ;; label = @13
                              local.get 3
                              i32.const 0
                              i32.lt_s
                              br_if 0 (;@13;)
                              local.get 1
                              f64.load offset=8
                              local.set 2
                              br 1 (;@12;)
                            end
                            i32.const 0
                            local.get 4
                            i32.sub
                            local.set 4
                            local.get 1
                            f64.load offset=8
                            f64.neg
                            local.set 2
                          end
                          local.get 4
                          i32.const 3
                          i32.and
                          br_table 2 (;@9;) 3 (;@8;) 4 (;@7;) 1 (;@10;) 2 (;@9;)
                        end
                        local.get 0
                        local.get 0
                        f32.sub
                        local.set 0
                        br 9 (;@1;)
                      end
                      local.get 2
                      local.get 2
                      local.get 2
                      f64.mul
                      local.tee 5
                      f64.mul
                      local.tee 7
                      local.get 5
                      local.get 5
                      f64.mul
                      f64.mul
                      local.get 5
                      f64.const 0x1.6cd878c3b46a7p-19 (;=0.000002718311493989822;)
                      f64.mul
                      f64.const -0x1.a00f9e2cae774p-13 (;=-0.00019839334836096632;)
                      f64.add
                      f64.mul
                      local.get 2
                      local.get 7
                      local.get 5
                      f64.const 0x1.11110896efbb2p-7 (;=0.008333329385889463;)
                      f64.mul
                      f64.const -0x1.5555554cbac77p-3 (;=-0.16666666641626524;)
                      f64.add
                      f64.mul
                      f64.add
                      f64.add
                      f32.demote_f64
                      local.set 0
                      br 8 (;@1;)
                    end
                    local.get 2
                    local.get 2
                    f64.mul
                    local.tee 2
                    f64.const -0x1.ffffffd0c5e81p-2 (;=-0.499999997251031;)
                    f64.mul
                    f64.const 0x1p+0 (;=1;)
                    f64.add
                    local.get 2
                    local.get 2
                    f64.mul
                    local.tee 5
                    f64.const 0x1.55553e1053a42p-5 (;=0.04166662332373906;)
                    f64.mul
                    f64.add
                    local.get 2
                    local.get 5
                    f64.mul
                    local.get 2
                    f64.const 0x1.99342e0ee5069p-16 (;=0.00002439044879627741;)
                    f64.mul
                    f64.const -0x1.6c087e80f1e27p-10 (;=-0.001388676377460993;)
                    f64.add
                    f64.mul
                    f64.add
                    f32.demote_f64
                    local.set 0
                    br 7 (;@1;)
                  end
                  local.get 2
                  local.get 2
                  f64.mul
                  local.tee 5
                  local.get 2
                  f64.neg
                  f64.mul
                  local.tee 7
                  local.get 5
                  local.get 5
                  f64.mul
                  f64.mul
                  local.get 5
                  f64.const 0x1.6cd878c3b46a7p-19 (;=0.000002718311493989822;)
                  f64.mul
                  f64.const -0x1.a00f9e2cae774p-13 (;=-0.00019839334836096632;)
                  f64.add
                  f64.mul
                  local.get 7
                  local.get 5
                  f64.const 0x1.11110896efbb2p-7 (;=0.008333329385889463;)
                  f64.mul
                  f64.const -0x1.5555554cbac77p-3 (;=-0.16666666641626524;)
                  f64.add
                  f64.mul
                  local.get 2
                  f64.sub
                  f64.add
                  f32.demote_f64
                  local.set 0
                  br 6 (;@1;)
                end
                local.get 2
                local.get 2
                f64.mul
                local.tee 2
                f64.const -0x1.ffffffd0c5e81p-2 (;=-0.499999997251031;)
                f64.mul
                f64.const 0x1p+0 (;=1;)
                f64.add
                local.get 2
                local.get 2
                f64.mul
                local.tee 5
                f64.const 0x1.55553e1053a42p-5 (;=0.04166662332373906;)
                f64.mul
                f64.add
                local.get 2
                local.get 5
                f64.mul
                local.get 2
                f64.const 0x1.99342e0ee5069p-16 (;=0.00002439044879627741;)
                f64.mul
                f64.const -0x1.6c087e80f1e27p-10 (;=-0.001388676377460993;)
                f64.add
                f64.mul
                f64.add
                f32.demote_f64
                f32.neg
                local.set 0
                br 5 (;@1;)
              end
              local.get 4
              i32.const 1085271519
              i32.gt_u
              br_if 2 (;@3;)
              block ;; label = @6
                local.get 3
                i32.const -1
                i32.le_s
                br_if 0 (;@6;)
                local.get 2
                f64.const -0x1.2d97c7f3321d2p+2 (;=-4.71238898038469;)
                f64.add
                local.tee 5
                local.get 5
                local.get 5
                f64.mul
                local.tee 2
                f64.mul
                local.tee 7
                local.get 2
                local.get 2
                f64.mul
                f64.mul
                local.get 2
                f64.const 0x1.6cd878c3b46a7p-19 (;=0.000002718311493989822;)
                f64.mul
                f64.const -0x1.a00f9e2cae774p-13 (;=-0.00019839334836096632;)
                f64.add
                f64.mul
                local.get 5
                local.get 7
                local.get 2
                f64.const 0x1.11110896efbb2p-7 (;=0.008333329385889463;)
                f64.mul
                f64.const -0x1.5555554cbac77p-3 (;=-0.16666666641626524;)
                f64.add
                f64.mul
                f64.add
                f64.add
                f32.demote_f64
                local.set 0
                br 5 (;@1;)
              end
              f64.const -0x1.2d97c7f3321d2p+2 (;=-4.71238898038469;)
              local.get 2
              f64.sub
              local.tee 5
              local.get 5
              local.get 5
              f64.mul
              local.tee 2
              f64.mul
              local.tee 7
              local.get 2
              local.get 2
              f64.mul
              f64.mul
              local.get 2
              f64.const 0x1.6cd878c3b46a7p-19 (;=0.000002718311493989822;)
              f64.mul
              f64.const -0x1.a00f9e2cae774p-13 (;=-0.00019839334836096632;)
              f64.add
              f64.mul
              local.get 5
              local.get 7
              local.get 2
              f64.const 0x1.11110896efbb2p-7 (;=0.008333329385889463;)
              f64.mul
              f64.const -0x1.5555554cbac77p-3 (;=-0.16666666641626524;)
              f64.add
              f64.mul
              f64.add
              f64.add
              f32.demote_f64
              local.set 0
              br 4 (;@1;)
            end
            local.get 4
            i32.const 1075235811
            i32.gt_u
            br_if 2 (;@2;)
            block ;; label = @5
              local.get 3
              i32.const -1
              i32.le_s
              br_if 0 (;@5;)
              f64.const 0x1.921fb54442d18p+0 (;=1.5707963267948966;)
              local.get 2
              f64.sub
              local.tee 5
              local.get 5
              local.get 5
              f64.mul
              local.tee 2
              f64.mul
              local.tee 7
              local.get 2
              local.get 2
              f64.mul
              f64.mul
              local.get 2
              f64.const 0x1.6cd878c3b46a7p-19 (;=0.000002718311493989822;)
              f64.mul
              f64.const -0x1.a00f9e2cae774p-13 (;=-0.00019839334836096632;)
              f64.add
              f64.mul
              local.get 5
              local.get 7
              local.get 2
              f64.const 0x1.11110896efbb2p-7 (;=0.008333329385889463;)
              f64.mul
              f64.const -0x1.5555554cbac77p-3 (;=-0.16666666641626524;)
              f64.add
              f64.mul
              f64.add
              f64.add
              f32.demote_f64
              local.set 0
              br 4 (;@1;)
            end
            local.get 2
            f64.const 0x1.921fb54442d18p+0 (;=1.5707963267948966;)
            f64.add
            local.tee 5
            local.get 5
            local.get 5
            f64.mul
            local.tee 2
            f64.mul
            local.tee 7
            local.get 2
            local.get 2
            f64.mul
            f64.mul
            local.get 2
            f64.const 0x1.6cd878c3b46a7p-19 (;=0.000002718311493989822;)
            f64.mul
            f64.const -0x1.a00f9e2cae774p-13 (;=-0.00019839334836096632;)
            f64.add
            f64.mul
            local.get 5
            local.get 7
            local.get 2
            f64.const 0x1.11110896efbb2p-7 (;=0.008333329385889463;)
            f64.mul
            f64.const -0x1.5555554cbac77p-3 (;=-0.16666666641626524;)
            f64.add
            f64.mul
            f64.add
            f64.add
            f32.demote_f64
            local.set 0
            br 3 (;@1;)
          end
          block ;; label = @4
            local.get 4
            i32.const 964689920
            i32.lt_u
            br_if 0 (;@4;)
            local.get 2
            local.get 2
            f64.mul
            local.tee 2
            f64.const -0x1.ffffffd0c5e81p-2 (;=-0.499999997251031;)
            f64.mul
            f64.const 0x1p+0 (;=1;)
            f64.add
            local.get 2
            local.get 2
            f64.mul
            local.tee 5
            f64.const 0x1.55553e1053a42p-5 (;=0.04166662332373906;)
            f64.mul
            f64.add
            local.get 2
            local.get 5
            f64.mul
            local.get 2
            f64.const 0x1.99342e0ee5069p-16 (;=0.00002439044879627741;)
            f64.mul
            f64.const -0x1.6c087e80f1e27p-10 (;=-0.001388676377460993;)
            f64.add
            f64.mul
            f64.add
            f32.demote_f64
            local.set 0
            br 3 (;@1;)
          end
          local.get 1
          local.get 0
          f32.const 0x1p+120 (;=1329228000000000000000000000000000000;)
          f32.add
          f32.store offset=4
          local.get 1
          f32.load offset=4
          drop
          f32.const 0x1p+0 (;=1;)
          local.set 0
          br 2 (;@1;)
        end
        f64.const -0x1.921fb54442d18p+2 (;=-6.283185307179586;)
        f64.const 0x1.921fb54442d18p+2 (;=6.283185307179586;)
        local.get 3
        i32.const -1
        i32.gt_s
        select
        local.get 2
        f64.add
        local.tee 2
        local.get 2
        f64.mul
        local.tee 2
        f64.const -0x1.ffffffd0c5e81p-2 (;=-0.499999997251031;)
        f64.mul
        f64.const 0x1p+0 (;=1;)
        f64.add
        local.get 2
        local.get 2
        f64.mul
        local.tee 5
        f64.const 0x1.55553e1053a42p-5 (;=0.04166662332373906;)
        f64.mul
        f64.add
        local.get 2
        local.get 5
        f64.mul
        local.get 2
        f64.const 0x1.99342e0ee5069p-16 (;=0.00002439044879627741;)
        f64.mul
        f64.const -0x1.6c087e80f1e27p-10 (;=-0.001388676377460993;)
        f64.add
        f64.mul
        f64.add
        f32.demote_f64
        local.set 0
        br 1 (;@1;)
      end
      f64.const -0x1.921fb54442d18p+1 (;=-3.141592653589793;)
      f64.const 0x1.921fb54442d18p+1 (;=3.141592653589793;)
      local.get 3
      i32.const -1
      i32.gt_s
      select
      local.get 2
      f64.add
      local.tee 2
      local.get 2
      f64.mul
      local.tee 2
      f64.const -0x1.ffffffd0c5e81p-2 (;=-0.499999997251031;)
      f64.mul
      f64.const 0x1p+0 (;=1;)
      f64.add
      local.get 2
      local.get 2
      f64.mul
      local.tee 5
      f64.const 0x1.55553e1053a42p-5 (;=0.04166662332373906;)
      f64.mul
      f64.add
      local.get 2
      local.get 5
      f64.mul
      local.get 2
      f64.const 0x1.99342e0ee5069p-16 (;=0.00002439044879627741;)
      f64.mul
      f64.const -0x1.6c087e80f1e27p-10 (;=-0.001388676377460993;)
      f64.add
      f64.mul
      f64.add
      f32.demote_f64
      f32.neg
      local.set 0
    end
    local.get 1
    i32.const 16
    i32.add
    global.set 0
    local.get 0
  )
  (func (;13;) (type 1) (param i32 i32 i32 i32 i32)
    (local i32 i32 i32 i32 i32 i32)
    block ;; label = @1
      local.get 3
      i32.const 4
      i32.add
      local.get 4
      i32.gt_u
      br_if 0 (;@1;)
      local.get 2
      local.get 3
      i32.const 2
      i32.shl
      local.tee 5
      i32.add
      local.set 6
      local.get 1
      local.get 5
      i32.add
      local.set 7
      local.get 0
      local.get 5
      i32.add
      local.set 5
      loop ;; label = @2
        local.get 6
        local.get 5
        v128.load align=1
        local.get 7
        v128.load align=1
        f32x4.div
        v128.store align=1
        local.get 6
        i32.const 16
        i32.add
        local.set 6
        local.get 7
        i32.const 16
        i32.add
        local.set 7
        local.get 5
        i32.const 16
        i32.add
        local.set 5
        local.get 3
        local.tee 8
        i32.const 4
        i32.add
        local.set 3
        local.get 8
        i32.const 8
        i32.add
        local.get 4
        i32.le_u
        br_if 0 (;@2;)
      end
    end
    block ;; label = @1
      local.get 4
      local.get 3
      i32.le_u
      br_if 0 (;@1;)
      block ;; label = @2
        local.get 4
        local.get 3
        i32.sub
        local.tee 9
        i32.const 8
        i32.lt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 2
        i32.sub
        i32.const -16
        i32.gt_u
        br_if 0 (;@2;)
        local.get 1
        local.get 2
        i32.sub
        i32.const -16
        i32.gt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 3
        i32.const 2
        i32.shl
        local.tee 5
        i32.add
        local.set 6
        local.get 1
        local.get 5
        i32.add
        local.set 7
        local.get 2
        local.get 5
        i32.add
        local.set 5
        local.get 3
        local.get 9
        i32.const -4
        i32.and
        local.tee 10
        i32.add
        local.set 3
        local.get 10
        local.set 8
        loop ;; label = @3
          local.get 5
          local.get 6
          v128.load align=4
          local.get 7
          v128.load align=4
          f32x4.div
          v128.store align=4
          local.get 6
          i32.const 16
          i32.add
          local.set 6
          local.get 7
          i32.const 16
          i32.add
          local.set 7
          local.get 5
          i32.const 16
          i32.add
          local.set 5
          local.get 8
          i32.const -4
          i32.add
          local.tee 8
          br_if 0 (;@3;)
        end
        local.get 9
        local.get 10
        i32.eq
        br_if 1 (;@1;)
      end
      local.get 3
      i32.const 1
      i32.add
      local.set 6
      block ;; label = @2
        local.get 4
        local.get 3
        i32.sub
        i32.const 1
        i32.and
        i32.eqz
        br_if 0 (;@2;)
        local.get 2
        local.get 3
        i32.const 2
        i32.shl
        local.tee 7
        i32.add
        local.get 0
        local.get 7
        i32.add
        f32.load
        local.get 1
        local.get 7
        i32.add
        f32.load
        f32.div
        f32.store
        local.get 6
        local.set 3
      end
      local.get 4
      local.get 6
      i32.eq
      br_if 0 (;@1;)
      local.get 4
      local.get 3
      i32.sub
      local.set 8
      local.get 0
      local.get 3
      i32.const 2
      i32.shl
      local.tee 5
      i32.add
      local.set 6
      local.get 1
      local.get 5
      i32.add
      local.set 7
      local.get 2
      local.get 5
      i32.add
      local.set 5
      loop ;; label = @2
        local.get 5
        local.get 6
        f32.load
        local.get 7
        f32.load
        f32.div
        f32.store
        local.get 5
        i32.const 4
        i32.add
        local.get 6
        i32.const 4
        i32.add
        f32.load
        local.get 7
        i32.const 4
        i32.add
        f32.load
        f32.div
        f32.store
        local.get 6
        i32.const 8
        i32.add
        local.set 6
        local.get 7
        i32.const 8
        i32.add
        local.set 7
        local.get 5
        i32.const 8
        i32.add
        local.set 5
        local.get 8
        i32.const -2
        i32.add
        local.tee 8
        br_if 0 (;@2;)
      end
    end
  )
  (func (;14;) (type 8) (param i32 i32 i32 i32 i32 i32)
    (local i32)
    block ;; label = @1
      block ;; label = @2
        local.get 4
        local.get 5
        i32.ge_u
        br_if 0 (;@2;)
        local.get 3
        i32.eqz
        br_if 1 (;@1;)
        local.get 2
        local.get 4
        i32.const 2
        i32.shl
        i32.add
        local.set 2
        loop ;; label = @3
          local.get 2
          local.get 0
          local.get 1
          local.get 4
          local.get 3
          i32.div_u
          local.tee 6
          i32.const 2
          i32.shl
          i32.add
          f32.load
          i32.trunc_sat_f32_u
          local.get 3
          i32.mul
          i32.const 2
          i32.shl
          i32.add
          local.get 4
          local.get 6
          local.get 3
          i32.mul
          i32.sub
          i32.const 2
          i32.shl
          i32.add
          f32.load
          f32.store
          local.get 2
          i32.const 4
          i32.add
          local.set 2
          local.get 5
          local.get 4
          i32.const 1
          i32.add
          local.tee 4
          i32.ne
          br_if 0 (;@3;)
        end
      end
      return
    end
    call 3
    unreachable
  )
  (func (;15;) (type 8) (param i32 i32 i32 i32 i32 i32)
    (local i32)
    block ;; label = @1
      block ;; label = @2
        local.get 4
        local.get 5
        i32.ge_u
        br_if 0 (;@2;)
        local.get 3
        i32.eqz
        br_if 1 (;@1;)
        local.get 2
        local.get 4
        i32.const 2
        i32.shl
        i32.add
        local.set 2
        loop ;; label = @3
          local.get 0
          local.get 1
          local.get 4
          local.get 3
          i32.div_u
          local.tee 6
          i32.const 2
          i32.shl
          i32.add
          f32.load
          i32.trunc_sat_f32_u
          local.get 3
          i32.mul
          i32.const 2
          i32.shl
          i32.add
          local.get 4
          local.get 6
          local.get 3
          i32.mul
          i32.sub
          i32.const 2
          i32.shl
          i32.add
          local.tee 6
          local.get 2
          f32.load
          local.get 6
          f32.load
          f32.add
          f32.store
          local.get 2
          i32.const 4
          i32.add
          local.set 2
          local.get 5
          local.get 4
          i32.const 1
          i32.add
          local.tee 4
          i32.ne
          br_if 0 (;@3;)
        end
      end
      return
    end
    call 3
    unreachable
  )
  (func (;16;) (type 6) (param i32 i32 i32 i32)
    (local i32)
    block ;; label = @1
      local.get 3
      local.get 2
      i32.le_u
      br_if 0 (;@1;)
      local.get 3
      local.get 2
      i32.sub
      local.set 4
      local.get 0
      local.get 2
      i32.const 2
      i32.shl
      local.tee 3
      i32.add
      local.set 2
      local.get 1
      local.get 3
      i32.add
      local.set 3
      loop ;; label = @2
        local.get 3
        local.get 2
        f32.load
        call 17
        f32.store
        local.get 2
        i32.const 4
        i32.add
        local.set 2
        local.get 3
        i32.const 4
        i32.add
        local.set 3
        local.get 4
        i32.const -1
        i32.add
        local.tee 4
        br_if 0 (;@2;)
      end
    end
  )
  (func (;17;) (type 7) (param f32) (result f32)
    (local i32 i32 i32 i32 f32 f32 f32)
    global.get 0
    i32.const 16
    i32.sub
    local.set 1
    local.get 0
    i32.reinterpret_f32
    local.tee 2
    i32.const 31
    i32.shr_u
    local.set 3
    block ;; label = @1
      block ;; label = @2
        block ;; label = @3
          block ;; label = @4
            block ;; label = @5
              block ;; label = @6
                block ;; label = @7
                  local.get 2
                  i32.const 2147483647
                  i32.and
                  local.tee 4
                  i32.const 1118743632
                  i32.lt_u
                  br_if 0 (;@7;)
                  block ;; label = @8
                    local.get 4
                    i32.const 2139095040
                    i32.le_u
                    br_if 0 (;@8;)
                    local.get 0
                    return
                  end
                  block ;; label = @8
                    local.get 4
                    i32.const 1118925335
                    i32.gt_u
                    br_if 0 (;@8;)
                    local.get 2
                    i32.const -1
                    i32.gt_s
                    br_if 2 (;@6;)
                    local.get 1
                    f32.const -0x1p-126 (;=-0.000000000000000000000000000000000000011754944;)
                    local.get 0
                    f32.div
                    f32.store offset=8
                    local.get 1
                    f32.load offset=8
                    drop
                    br 2 (;@6;)
                  end
                  block ;; label = @8
                    local.get 2
                    i32.const -1
                    i32.gt_s
                    br_if 0 (;@8;)
                    local.get 1
                    f32.const -0x1p-126 (;=-0.000000000000000000000000000000000000011754944;)
                    local.get 0
                    f32.div
                    f32.store offset=8
                    local.get 1
                    f32.load offset=8
                    drop
                    f32.const 0x0p+0 (;=0;)
                    local.set 5
                    local.get 4
                    i32.const 1120924084
                    i32.le_u
                    br_if 2 (;@6;)
                    br 7 (;@1;)
                  end
                  local.get 0
                  f32.const 0x1p+127 (;=170141180000000000000000000000000000000;)
                  f32.mul
                  return
                end
                block ;; label = @7
                  local.get 4
                  i32.const 1051816472
                  i32.gt_u
                  br_if 0 (;@7;)
                  local.get 4
                  i32.const 956301312
                  i32.le_u
                  br_if 2 (;@5;)
                  i32.const 0
                  local.set 4
                  f32.const 0x0p+0 (;=0;)
                  local.set 6
                  local.get 0
                  local.set 5
                  br 5 (;@2;)
                end
                local.get 4
                i32.const 1065686418
                i32.le_u
                br_if 2 (;@4;)
              end
              local.get 0
              f32.const 0x1.715476p+0 (;=1.442695;)
              f32.mul
              local.get 3
              i32.const 2
              i32.shl
              f32.load offset=1048840
              f32.add
              i32.trunc_sat_f32_s
              local.set 4
              br 2 (;@3;)
            end
            local.get 1
            local.get 0
            f32.const 0x1p+127 (;=170141180000000000000000000000000000000;)
            f32.add
            f32.store offset=12
            local.get 1
            f32.load offset=12
            drop
            local.get 0
            f32.const 0x1p+0 (;=1;)
            f32.add
            return
          end
          local.get 3
          i32.const 1
          i32.xor
          local.get 3
          i32.sub
          local.set 4
        end
        local.get 0
        local.get 4
        f32.convert_i32_s
        local.tee 5
        f32.const -0x1.62e4p-1 (;=-0.69314575;)
        f32.mul
        f32.add
        local.tee 0
        local.get 5
        f32.const 0x1.7f7d1cp-20 (;=0.0000014286068;)
        f32.mul
        local.tee 6
        f32.sub
        local.set 5
      end
      local.get 0
      local.get 5
      local.get 5
      local.get 5
      local.get 5
      f32.mul
      local.tee 7
      local.get 7
      f32.const -0x1.6aa42ap-9 (;=-0.0027667333;)
      f32.mul
      f32.const 0x1.55551ep-3 (;=0.16666625;)
      f32.add
      f32.mul
      f32.sub
      local.tee 7
      f32.mul
      f32.const 0x1p+1 (;=2;)
      local.get 7
      f32.sub
      f32.div
      local.get 6
      f32.sub
      f32.add
      f32.const 0x1p+0 (;=1;)
      f32.add
      local.set 5
      local.get 4
      i32.eqz
      br_if 0 (;@1;)
      block ;; label = @2
        block ;; label = @3
          block ;; label = @4
            block ;; label = @5
              local.get 4
              i32.const 127
              i32.gt_s
              br_if 0 (;@5;)
              local.get 4
              i32.const -126
              i32.ge_s
              br_if 3 (;@2;)
              local.get 5
              f32.const 0x1p-102 (;=0.00000000000000000000000000000019721523;)
              f32.mul
              local.set 5
              local.get 4
              i32.const -229
              i32.le_u
              br_if 1 (;@4;)
              local.get 4
              i32.const 102
              i32.add
              local.set 4
              br 3 (;@2;)
            end
            local.get 5
            f32.const 0x1p+127 (;=170141180000000000000000000000000000000;)
            f32.mul
            local.set 5
            local.get 4
            i32.const 254
            i32.gt_u
            br_if 1 (;@3;)
            local.get 4
            i32.const -127
            i32.add
            local.set 4
            br 2 (;@2;)
          end
          local.get 5
          f32.const 0x1p-102 (;=0.00000000000000000000000000000019721523;)
          f32.mul
          local.set 5
          local.get 4
          i32.const -330
          local.get 4
          i32.const -330
          i32.gt_u
          select
          i32.const 204
          i32.add
          local.set 4
          br 1 (;@2;)
        end
        local.get 5
        f32.const 0x1p+127 (;=170141180000000000000000000000000000000;)
        f32.mul
        local.set 5
        local.get 4
        i32.const 381
        local.get 4
        i32.const 381
        i32.lt_u
        select
        i32.const -254
        i32.add
        local.set 4
      end
      local.get 5
      local.get 4
      i32.const 23
      i32.shl
      i32.const 1065353216
      i32.add
      i32.const 2139095040
      i32.and
      f32.reinterpret_i32
      f32.mul
      local.set 5
    end
    local.get 5
  )
  (func (;18;) (type 9) (param i32 f32 i32 i32)
    (local v128 i32 i32 i32 i32)
    local.get 1
    f32x4.splat
    local.set 4
    block ;; label = @1
      local.get 2
      i32.const 4
      i32.add
      local.get 3
      i32.gt_u
      br_if 0 (;@1;)
      local.get 0
      local.get 2
      i32.const 2
      i32.shl
      i32.add
      local.set 5
      loop ;; label = @2
        local.get 5
        local.get 4
        v128.store align=1
        local.get 5
        i32.const 16
        i32.add
        local.set 5
        local.get 2
        local.tee 6
        i32.const 4
        i32.add
        local.set 2
        local.get 6
        i32.const 8
        i32.add
        local.get 3
        i32.le_u
        br_if 0 (;@2;)
      end
    end
    block ;; label = @1
      local.get 3
      local.get 2
      i32.le_u
      br_if 0 (;@1;)
      block ;; label = @2
        local.get 3
        local.get 2
        i32.sub
        local.tee 7
        i32.const 4
        i32.lt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 2
        i32.const 2
        i32.shl
        i32.add
        local.set 5
        local.get 2
        local.get 7
        i32.const -4
        i32.and
        local.tee 8
        i32.add
        local.set 2
        local.get 8
        local.set 6
        loop ;; label = @3
          local.get 5
          local.get 4
          v128.store align=4
          local.get 5
          i32.const 16
          i32.add
          local.set 5
          local.get 6
          i32.const -4
          i32.add
          local.tee 6
          br_if 0 (;@3;)
        end
        local.get 7
        local.get 8
        i32.eq
        br_if 1 (;@1;)
      end
      local.get 3
      local.get 2
      i32.sub
      local.set 6
      local.get 0
      local.get 2
      i32.const 2
      i32.shl
      i32.add
      local.set 5
      loop ;; label = @2
        local.get 5
        local.get 1
        f32.store
        local.get 5
        i32.const 4
        i32.add
        local.set 5
        local.get 6
        i32.const -1
        i32.add
        local.tee 6
        br_if 0 (;@2;)
      end
    end
  )
  (func (;19;) (type 6) (param i32 i32 i32 i32)
    (local i32 i32 f32 f32 f32)
    global.get 0
    i32.const 16
    i32.sub
    local.tee 4
    global.set 0
    block ;; label = @1
      local.get 3
      local.get 2
      i32.le_u
      br_if 0 (;@1;)
      local.get 3
      local.get 2
      i32.sub
      local.set 5
      local.get 0
      local.get 2
      i32.const 2
      i32.shl
      local.tee 3
      i32.add
      local.set 2
      local.get 1
      local.get 3
      i32.add
      local.set 3
      loop ;; label = @2
        local.get 2
        f32.load
        local.tee 6
        f32.const 0x1p-1 (;=0.5;)
        f32.mul
        local.set 7
        block ;; label = @3
          block ;; label = @4
            block ;; label = @5
              local.get 6
              local.get 6
              local.get 6
              local.get 6
              f32.const 0x1.6e4e26p-5 (;=0.044715;)
              f32.mul
              f32.mul
              f32.mul
              f32.add
              f32.const 0x1.988454p-1 (;=0.7978846;)
              f32.mul
              local.tee 8
              f32.abs
              local.tee 6
              i32.reinterpret_f32
              local.tee 1
              i32.const 1057791828
              i32.gt_u
              br_if 0 (;@5;)
              local.get 1
              i32.const 1048757624
              i32.gt_u
              br_if 1 (;@4;)
              block ;; label = @6
                local.get 1
                i32.const 8388607
                i32.gt_u
                br_if 0 (;@6;)
                local.get 4
                local.get 8
                local.get 8
                f32.mul
                f32.store offset=12
                local.get 4
                f32.load offset=12
                drop
                br 3 (;@3;)
              end
              local.get 6
              f32.const -0x1p+1 (;=-2;)
              f32.mul
              call 20
              local.tee 6
              f32.neg
              local.get 6
              f32.const 0x1p+1 (;=2;)
              f32.add
              f32.div
              local.set 6
              br 2 (;@3;)
            end
            block ;; label = @5
              local.get 1
              i32.const 1092616192
              i32.gt_u
              br_if 0 (;@5;)
              f32.const 0x1p+0 (;=1;)
              f32.const 0x1p+1 (;=2;)
              local.get 6
              local.get 6
              f32.add
              call 20
              f32.const 0x1p+1 (;=2;)
              f32.add
              f32.div
              f32.sub
              local.set 6
              br 2 (;@3;)
            end
            f32.const 0x0p+0 (;=0;)
            local.get 6
            f32.div
            f32.const 0x1p+0 (;=1;)
            f32.add
            local.set 6
            br 1 (;@3;)
          end
          local.get 6
          local.get 6
          f32.add
          call 20
          local.tee 6
          local.get 6
          f32.const 0x1p+1 (;=2;)
          f32.add
          f32.div
          local.set 6
        end
        local.get 3
        local.get 7
        local.get 6
        f32.neg
        local.get 6
        local.get 8
        i32.reinterpret_f32
        i32.const 0
        i32.lt_s
        select
        f32.const 0x1p+0 (;=1;)
        f32.add
        f32.mul
        f32.store
        local.get 2
        i32.const 4
        i32.add
        local.set 2
        local.get 3
        i32.const 4
        i32.add
        local.set 3
        local.get 5
        i32.const -1
        i32.add
        local.tee 5
        br_if 0 (;@2;)
      end
    end
    local.get 4
    i32.const 16
    i32.add
    global.set 0
  )
  (func (;20;) (type 7) (param f32) (result f32)
    (local i32 i32 i32 f32 f32 f32 f32)
    global.get 0
    i32.const 16
    i32.sub
    local.set 1
    block ;; label = @1
      block ;; label = @2
        block ;; label = @3
          block ;; label = @4
            block ;; label = @5
              block ;; label = @6
                block ;; label = @7
                  block ;; label = @8
                    block ;; label = @9
                      local.get 0
                      i32.reinterpret_f32
                      local.tee 2
                      i32.const 2147483647
                      i32.and
                      local.tee 3
                      i32.const 1100331075
                      i32.gt_u
                      br_if 0 (;@9;)
                      local.get 3
                      i32.const 1051816472
                      i32.gt_u
                      br_if 1 (;@8;)
                      local.get 3
                      i32.const 855638016
                      i32.lt_u
                      br_if 6 (;@3;)
                      i32.const 0
                      local.set 3
                      f32.const 0x0p+0 (;=0;)
                      local.set 4
                      br 5 (;@4;)
                    end
                    local.get 0
                    f32.const -0x1p+0 (;=-1;)
                    local.get 3
                    i32.const 2139095040
                    i32.gt_u
                    local.tee 1
                    select
                    local.set 5
                    local.get 2
                    i32.const 0
                    i32.lt_s
                    br_if 7 (;@1;)
                    local.get 1
                    br_if 7 (;@1;)
                    f32.const 0x1p-1 (;=0.5;)
                    local.set 5
                    local.get 3
                    i32.const 1118925336
                    i32.lt_u
                    br_if 1 (;@7;)
                    local.get 0
                    f32.const 0x1p+127 (;=170141180000000000000000000000000000000;)
                    f32.mul
                    return
                  end
                  local.get 3
                  i32.const 1065686418
                  i32.lt_u
                  br_if 1 (;@6;)
                  f32.const -0x1p-1 (;=-0.5;)
                  f32.const 0x1p-1 (;=0.5;)
                  local.get 2
                  i32.const 0
                  i32.lt_s
                  select
                  local.set 5
                end
                local.get 0
                f32.const 0x1.715476p+0 (;=1.442695;)
                f32.mul
                local.get 5
                f32.add
                i32.trunc_sat_f32_s
                local.tee 3
                f32.convert_i32_s
                local.tee 4
                f32.const 0x1.2fefa2p-17 (;=0.000009058001;)
                f32.mul
                local.set 5
                local.get 0
                local.get 4
                f32.const -0x1.62e3p-1 (;=-0.6931381;)
                f32.mul
                f32.add
                local.set 4
                br 1 (;@5;)
              end
              block ;; label = @6
                local.get 2
                i32.const 0
                i32.lt_s
                br_if 0 (;@6;)
                local.get 0
                f32.const -0x1.62e3p-1 (;=-0.6931381;)
                f32.add
                local.set 4
                f32.const 0x1.2fefa2p-17 (;=0.000009058001;)
                local.set 5
                i32.const 1
                local.set 3
                br 1 (;@5;)
              end
              local.get 0
              f32.const 0x1.62e3p-1 (;=0.6931381;)
              f32.add
              local.set 4
              f32.const -0x1.2fefa2p-17 (;=-0.000009058001;)
              local.set 5
              i32.const -1
              local.set 3
            end
            local.get 4
            local.get 4
            local.get 5
            f32.sub
            local.tee 0
            f32.sub
            local.get 5
            f32.sub
            local.set 4
          end
          local.get 0
          local.get 0
          f32.const 0x1p-1 (;=0.5;)
          f32.mul
          local.tee 6
          f32.mul
          local.tee 5
          local.get 5
          local.get 5
          f32.const 0x1.9e602p-10 (;=0.001580717;)
          f32.mul
          f32.const -0x1.1110dp-5 (;=-0.033333212;)
          f32.add
          f32.mul
          f32.const 0x1p+0 (;=1;)
          f32.add
          local.tee 7
          f32.const 0x1.8p+1 (;=3;)
          local.get 6
          local.get 7
          f32.mul
          f32.sub
          local.tee 6
          f32.sub
          f32.const 0x1.8p+2 (;=6;)
          local.get 0
          local.get 6
          f32.mul
          f32.sub
          f32.div
          f32.mul
          local.set 6
          local.get 3
          br_if 1 (;@2;)
          local.get 0
          local.get 0
          local.get 6
          f32.mul
          local.get 5
          f32.sub
          f32.sub
          return
        end
        block ;; label = @3
          local.get 3
          i32.const 8388608
          i32.lt_u
          br_if 0 (;@3;)
          local.get 0
          return
        end
        local.get 1
        local.get 0
        local.get 0
        f32.mul
        f32.store offset=12
        local.get 1
        f32.load offset=12
        drop
        local.get 0
        return
      end
      local.get 0
      local.get 6
      local.get 4
      f32.sub
      f32.mul
      local.get 4
      f32.sub
      local.get 5
      f32.sub
      local.set 5
      block ;; label = @2
        block ;; label = @3
          block ;; label = @4
            local.get 3
            i32.const 1
            i32.add
            br_table 0 (;@4;) 2 (;@2;) 1 (;@3;) 2 (;@2;)
          end
          local.get 0
          local.get 5
          f32.sub
          f32.const 0x1p-1 (;=0.5;)
          f32.mul
          f32.const -0x1p-1 (;=-0.5;)
          f32.add
          return
        end
        block ;; label = @3
          local.get 0
          f32.const -0x1p-2 (;=-0.25;)
          f32.lt
          br_if 0 (;@3;)
          local.get 0
          local.get 5
          f32.sub
          local.tee 0
          local.get 0
          f32.add
          f32.const 0x1p+0 (;=1;)
          f32.add
          return
        end
        local.get 5
        local.get 0
        f32.const 0x1p-1 (;=0.5;)
        f32.add
        f32.sub
        f32.const -0x1p+1 (;=-2;)
        f32.mul
        return
      end
      local.get 3
      i32.const 23
      i32.shl
      local.tee 2
      i32.const 1065353216
      i32.add
      f32.reinterpret_i32
      local.set 4
      block ;; label = @2
        local.get 3
        i32.const 57
        i32.lt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 5
        f32.sub
        f32.const 0x1p+0 (;=1;)
        f32.add
        local.tee 0
        local.get 0
        f32.add
        f32.const 0x1p+127 (;=170141180000000000000000000000000000000;)
        f32.mul
        local.get 0
        local.get 4
        f32.mul
        local.get 3
        i32.const 128
        i32.eq
        select
        f32.const -0x1p+0 (;=-1;)
        f32.add
        return
      end
      i32.const 1065353216
      local.get 2
      i32.sub
      f32.reinterpret_i32
      local.set 6
      block ;; label = @2
        block ;; label = @3
          local.get 3
          i32.const 23
          i32.lt_u
          br_if 0 (;@3;)
          local.get 0
          local.get 5
          local.get 6
          f32.add
          f32.sub
          f32.const 0x1p+0 (;=1;)
          f32.add
          local.set 0
          br 1 (;@2;)
        end
        f32.const 0x1p+0 (;=1;)
        local.get 6
        f32.sub
        local.get 0
        local.get 5
        f32.sub
        f32.add
        local.set 0
      end
      local.get 0
      local.get 4
      f32.mul
      local.set 5
    end
    local.get 5
  )
  (func (;21;) (type 1) (param i32 i32 i32 i32 i32)
    (local i32 i32 i32 f32 f32 f32 f32)
    global.get 0
    i32.const 16
    i32.sub
    local.tee 5
    global.set 0
    block ;; label = @1
      local.get 4
      local.get 3
      i32.le_u
      br_if 0 (;@1;)
      local.get 4
      local.get 3
      i32.sub
      local.set 6
      local.get 0
      local.get 3
      i32.const 2
      i32.shl
      local.tee 7
      i32.add
      local.set 3
      local.get 1
      local.get 7
      i32.add
      local.set 4
      local.get 2
      local.get 7
      i32.add
      local.set 2
      loop ;; label = @2
        block ;; label = @3
          block ;; label = @4
            block ;; label = @5
              local.get 3
              f32.load
              local.tee 8
              local.get 8
              f32.const 0x1.6e4e26p-5 (;=0.044715;)
              f32.mul
              local.get 8
              local.get 8
              f32.mul
              local.tee 9
              f32.mul
              f32.add
              f32.const 0x1.988454p-1 (;=0.7978846;)
              f32.mul
              local.tee 10
              f32.abs
              local.tee 11
              i32.reinterpret_f32
              local.tee 1
              i32.const 1057791828
              i32.gt_u
              br_if 0 (;@5;)
              local.get 1
              i32.const 1048757624
              i32.gt_u
              br_if 1 (;@4;)
              block ;; label = @6
                local.get 1
                i32.const 8388607
                i32.gt_u
                br_if 0 (;@6;)
                local.get 5
                local.get 10
                local.get 10
                f32.mul
                f32.store offset=12
                local.get 5
                f32.load offset=12
                drop
                br 3 (;@3;)
              end
              local.get 11
              f32.const -0x1p+1 (;=-2;)
              f32.mul
              call 20
              local.tee 11
              f32.neg
              local.get 11
              f32.const 0x1p+1 (;=2;)
              f32.add
              f32.div
              local.set 11
              br 2 (;@3;)
            end
            block ;; label = @5
              local.get 1
              i32.const 1092616192
              i32.gt_u
              br_if 0 (;@5;)
              f32.const 0x1p+0 (;=1;)
              f32.const 0x1p+1 (;=2;)
              local.get 11
              local.get 11
              f32.add
              call 20
              f32.const 0x1p+1 (;=2;)
              f32.add
              f32.div
              f32.sub
              local.set 11
              br 2 (;@3;)
            end
            f32.const 0x0p+0 (;=0;)
            local.get 11
            f32.div
            f32.const 0x1p+0 (;=1;)
            f32.add
            local.set 11
            br 1 (;@3;)
          end
          local.get 11
          local.get 11
          f32.add
          call 20
          local.tee 11
          local.get 11
          f32.const 0x1p+1 (;=2;)
          f32.add
          f32.div
          local.set 11
        end
        local.get 2
        local.get 4
        f32.load
        local.get 11
        f32.neg
        local.get 11
        local.get 10
        i32.reinterpret_f32
        i32.const 0
        i32.lt_s
        select
        local.tee 11
        f32.const 0x1p+0 (;=1;)
        f32.add
        f32.const 0x1p-1 (;=0.5;)
        f32.mul
        local.get 8
        f32.const 0x1p-1 (;=0.5;)
        f32.mul
        f32.const 0x1p+0 (;=1;)
        local.get 11
        local.get 11
        f32.mul
        f32.sub
        f32.mul
        local.get 9
        f32.const 0x1.12ba9cp-3 (;=0.13414499;)
        f32.mul
        f32.const 0x1p+0 (;=1;)
        f32.add
        f32.const 0x1.988454p-1 (;=0.7978846;)
        f32.mul
        f32.mul
        f32.add
        f32.mul
        f32.store
        local.get 3
        i32.const 4
        i32.add
        local.set 3
        local.get 4
        i32.const 4
        i32.add
        local.set 4
        local.get 2
        i32.const 4
        i32.add
        local.set 2
        local.get 6
        i32.const -1
        i32.add
        local.tee 6
        br_if 0 (;@2;)
      end
    end
    local.get 5
    i32.const 16
    i32.add
    global.set 0
  )
  (func (;22;) (type 10) (param i32 i32 f32 i32 i32)
    (local i32 i32 i32 i32 v128 i32 v128 f32)
    block ;; label = @1
      local.get 4
      local.get 3
      i32.le_u
      br_if 0 (;@1;)
      block ;; label = @2
        local.get 4
        local.get 3
        i32.sub
        local.tee 5
        i32.const 4
        i32.lt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 1
        i32.sub
        i32.const -16
        i32.gt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 3
        i32.const 2
        i32.shl
        local.tee 6
        i32.add
        local.set 7
        local.get 1
        local.get 6
        i32.add
        local.set 6
        local.get 3
        local.get 5
        i32.const -4
        i32.and
        local.tee 8
        i32.add
        local.set 3
        local.get 2
        f32x4.splat
        local.set 9
        local.get 8
        local.set 10
        loop ;; label = @3
          local.get 6
          local.get 7
          v128.load align=4
          local.tee 11
          local.get 9
          local.get 11
          f32x4.mul
          local.get 11
          v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
          f32x4.ge
          v128.bitselect
          v128.store align=4
          local.get 7
          i32.const 16
          i32.add
          local.set 7
          local.get 6
          i32.const 16
          i32.add
          local.set 6
          local.get 10
          i32.const -4
          i32.add
          local.tee 10
          br_if 0 (;@3;)
        end
        local.get 5
        local.get 8
        i32.eq
        br_if 1 (;@1;)
      end
      local.get 3
      i32.const 1
      i32.add
      local.set 7
      block ;; label = @2
        local.get 4
        local.get 3
        i32.sub
        i32.const 1
        i32.and
        i32.eqz
        br_if 0 (;@2;)
        local.get 1
        local.get 3
        i32.const 2
        i32.shl
        local.tee 6
        i32.add
        local.get 0
        local.get 6
        i32.add
        f32.load
        local.tee 12
        local.get 2
        local.get 12
        f32.mul
        local.get 12
        f32.const 0x0p+0 (;=0;)
        f32.ge
        select
        f32.store
        local.get 7
        local.set 3
      end
      local.get 4
      local.get 7
      i32.eq
      br_if 0 (;@1;)
      local.get 4
      local.get 3
      i32.sub
      local.set 10
      local.get 0
      local.get 3
      i32.const 2
      i32.shl
      local.tee 6
      i32.add
      local.set 7
      local.get 1
      local.get 6
      i32.add
      local.set 6
      loop ;; label = @2
        local.get 6
        local.get 7
        f32.load
        local.tee 12
        local.get 2
        local.get 12
        f32.mul
        local.get 12
        f32.const 0x0p+0 (;=0;)
        f32.ge
        select
        f32.store
        local.get 6
        i32.const 4
        i32.add
        local.get 7
        i32.const 4
        i32.add
        f32.load
        local.tee 12
        local.get 2
        local.get 12
        f32.mul
        local.get 12
        f32.const 0x0p+0 (;=0;)
        f32.ge
        select
        f32.store
        local.get 7
        i32.const 8
        i32.add
        local.set 7
        local.get 6
        i32.const 8
        i32.add
        local.set 6
        local.get 10
        i32.const -2
        i32.add
        local.tee 10
        br_if 0 (;@2;)
      end
    end
  )
  (func (;23;) (type 11) (param i32 i32 i32 f32 i32 i32)
    (local i32 i32 i32 i32 i32 v128 i32)
    block ;; label = @1
      local.get 5
      local.get 4
      i32.le_u
      br_if 0 (;@1;)
      block ;; label = @2
        local.get 5
        local.get 4
        i32.sub
        local.tee 6
        i32.const 8
        i32.lt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 2
        i32.sub
        i32.const -16
        i32.gt_u
        br_if 0 (;@2;)
        local.get 1
        local.get 2
        i32.sub
        i32.const -16
        i32.gt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 4
        i32.const 2
        i32.shl
        local.tee 7
        i32.add
        local.set 8
        local.get 1
        local.get 7
        i32.add
        local.set 9
        local.get 2
        local.get 7
        i32.add
        local.set 7
        local.get 4
        local.get 6
        i32.const -4
        i32.and
        local.tee 10
        i32.add
        local.set 4
        local.get 3
        f32x4.splat
        local.set 11
        local.get 10
        local.set 12
        loop ;; label = @3
          local.get 7
          local.get 9
          v128.load align=4
          v128.const i32x4 0x3f800000 0x3f800000 0x3f800000 0x3f800000
          local.get 11
          local.get 8
          v128.load align=4
          v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
          f32x4.ge
          v128.bitselect
          f32x4.mul
          v128.store align=4
          local.get 8
          i32.const 16
          i32.add
          local.set 8
          local.get 9
          i32.const 16
          i32.add
          local.set 9
          local.get 7
          i32.const 16
          i32.add
          local.set 7
          local.get 12
          i32.const -4
          i32.add
          local.tee 12
          br_if 0 (;@3;)
        end
        local.get 6
        local.get 10
        i32.eq
        br_if 1 (;@1;)
      end
      local.get 4
      i32.const 1
      i32.add
      local.set 8
      block ;; label = @2
        local.get 5
        local.get 4
        i32.sub
        i32.const 1
        i32.and
        i32.eqz
        br_if 0 (;@2;)
        local.get 2
        local.get 4
        i32.const 2
        i32.shl
        local.tee 9
        i32.add
        local.get 1
        local.get 9
        i32.add
        f32.load
        f32.const 0x1p+0 (;=1;)
        local.get 3
        local.get 0
        local.get 9
        i32.add
        f32.load
        f32.const 0x0p+0 (;=0;)
        f32.ge
        select
        f32.mul
        f32.store
        local.get 8
        local.set 4
      end
      local.get 5
      local.get 8
      i32.eq
      br_if 0 (;@1;)
      local.get 5
      local.get 4
      i32.sub
      local.set 12
      local.get 0
      local.get 4
      i32.const 2
      i32.shl
      local.tee 7
      i32.add
      local.set 8
      local.get 1
      local.get 7
      i32.add
      local.set 9
      local.get 2
      local.get 7
      i32.add
      local.set 7
      loop ;; label = @2
        local.get 7
        local.get 9
        f32.load
        f32.const 0x1p+0 (;=1;)
        local.get 3
        local.get 8
        f32.load
        f32.const 0x0p+0 (;=0;)
        f32.ge
        select
        f32.mul
        f32.store
        local.get 7
        i32.const 4
        i32.add
        local.get 9
        i32.const 4
        i32.add
        f32.load
        f32.const 0x1p+0 (;=1;)
        local.get 3
        local.get 8
        i32.const 4
        i32.add
        f32.load
        f32.const 0x0p+0 (;=0;)
        f32.ge
        select
        f32.mul
        f32.store
        local.get 8
        i32.const 8
        i32.add
        local.set 8
        local.get 9
        i32.const 8
        i32.add
        local.set 9
        local.get 7
        i32.const 8
        i32.add
        local.set 7
        local.get 12
        i32.const -2
        i32.add
        local.tee 12
        br_if 0 (;@2;)
      end
    end
  )
  (func (;24;) (type 6) (param i32 i32 i32 i32)
    (local i32 i32 i32 i32 i32 v128 v128 v128 v128 v128 v128 v128 f32 f32 f32)
    block ;; label = @1
      local.get 3
      local.get 2
      i32.le_u
      br_if 0 (;@1;)
      block ;; label = @2
        local.get 3
        local.get 2
        i32.sub
        local.tee 4
        i32.const 4
        i32.lt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 1
        i32.sub
        i32.const -16
        i32.gt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 2
        i32.const 2
        i32.shl
        local.tee 5
        i32.add
        local.set 6
        local.get 1
        local.get 5
        i32.add
        local.set 5
        local.get 2
        local.get 4
        i32.const -4
        i32.and
        local.tee 7
        i32.add
        local.set 2
        local.get 7
        local.set 8
        loop ;; label = @3
          local.get 5
          v128.const i32x4 0xff800000 0xff800000 0xff800000 0xff800000
          local.get 6
          v128.load align=4
          local.tee 9
          local.get 9
          v128.const i32x4 0x4c000000 0x4c000000 0x4c000000 0x4c000000
          f32x4.mul
          local.get 9
          v128.const i32x4 0x3f800000 0x3f800000 0x3f800000 0x3f800000
          i32x4.ne
          local.tee 10
          local.get 9
          v128.const i32x4 0xff800000 0xff800000 0xff800000 0xff800000
          i32x4.add
          v128.const i32x4 0x7f000000 0x7f000000 0x7f000000 0x7f000000
          i32x4.lt_u
          v128.and
          local.tee 11
          v128.bitselect
          v128.const i32x4 0x004afb0d 0x004afb0d 0x004afb0d 0x004afb0d
          i32x4.add
          local.tee 12
          i32.const 23
          i32x4.shr_u
          v128.const i32x4 0xffffff81 0xffffff81 0xffffff81 0xffffff81
          v128.const i32x4 0xffffff68 0xffffff68 0xffffff68 0xffffff68
          local.get 11
          v128.bitselect
          i32x4.add
          f32x4.convert_i32x4_s
          local.tee 13
          v128.const i32x4 0x3f317180 0x3f317180 0x3f317180 0x3f317180
          f32x4.mul
          local.get 12
          v128.const i32x4 0x007fffff 0x007fffff 0x007fffff 0x007fffff
          local.tee 14
          v128.and
          v128.const i32x4 0x3f3504f3 0x3f3504f3 0x3f3504f3 0x3f3504f3
          i32x4.add
          v128.const i32x4 0xbf800000 0xbf800000 0xbf800000 0xbf800000
          f32x4.add
          local.tee 12
          local.get 13
          v128.const i32x4 0x3717f7d1 0x3717f7d1 0x3717f7d1 0x3717f7d1
          f32x4.mul
          local.get 12
          local.get 12
          v128.const i32x4 0x40000000 0x40000000 0x40000000 0x40000000
          f32x4.add
          f32x4.div
          local.tee 13
          local.get 12
          local.get 12
          v128.const i32x4 0x3f000000 0x3f000000 0x3f000000 0x3f000000
          f32x4.mul
          f32x4.mul
          local.tee 15
          local.get 13
          local.get 13
          f32x4.mul
          local.tee 12
          local.get 12
          local.get 12
          f32x4.mul
          local.tee 12
          v128.const i32x4 0x3e91e9ee 0x3e91e9ee 0x3e91e9ee 0x3e91e9ee
          f32x4.mul
          v128.const i32x4 0x3f2aaaaa 0x3f2aaaaa 0x3f2aaaaa 0x3f2aaaaa
          f32x4.add
          f32x4.mul
          local.get 12
          local.get 12
          v128.const i32x4 0x3e789e26 0x3e789e26 0x3e789e26 0x3e789e26
          f32x4.mul
          v128.const i32x4 0x3eccce13 0x3eccce13 0x3eccce13 0x3eccce13
          f32x4.add
          f32x4.mul
          f32x4.add
          f32x4.add
          f32x4.mul
          f32x4.add
          local.get 15
          f32x4.sub
          f32x4.add
          f32x4.add
          local.get 10
          local.get 9
          v128.and
          local.get 9
          v128.const i32x4 0xffffffff 0xffffffff 0xffffffff 0xffffffff
          i32x4.gt_s
          local.tee 13
          local.get 9
          local.get 14
          i32x4.gt_s
          local.get 9
          v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
          local.tee 10
          f32x4.eq
          local.tee 12
          v128.or
          v128.andnot
          local.get 11
          v128.or
          v128.bitselect
          local.get 9
          local.get 14
          i32x4.le_s
          local.get 12
          v128.and
          v128.bitselect
          local.get 9
          local.get 9
          f32x4.sub
          local.get 10
          f32x4.div
          local.get 13
          local.get 12
          v128.or
          v128.bitselect
          v128.store align=4
          local.get 6
          i32.const 16
          i32.add
          local.set 6
          local.get 5
          i32.const 16
          i32.add
          local.set 5
          local.get 8
          i32.const -4
          i32.add
          local.tee 8
          br_if 0 (;@3;)
        end
        local.get 4
        local.get 7
        i32.eq
        br_if 1 (;@1;)
      end
      local.get 3
      local.get 2
      i32.sub
      local.set 3
      local.get 0
      local.get 2
      i32.const 2
      i32.shl
      local.tee 5
      i32.add
      local.set 6
      local.get 1
      local.get 5
      i32.add
      local.set 5
      loop ;; label = @2
        block ;; label = @3
          block ;; label = @4
            block ;; label = @5
              block ;; label = @6
                local.get 6
                f32.load
                local.tee 16
                i32.reinterpret_f32
                local.tee 8
                i32.const 8388608
                i32.lt_s
                br_if 0 (;@6;)
                local.get 8
                i32.const 2139095039
                i32.gt_u
                br_if 3 (;@3;)
                i32.const -127
                local.set 2
                local.get 8
                i32.const 1065353216
                i32.ne
                br_if 1 (;@5;)
                f32.const 0x0p+0 (;=0;)
                local.set 16
                br 3 (;@3;)
              end
              block ;; label = @6
                local.get 16
                f32.const 0x0p+0 (;=0;)
                f32.ne
                br_if 0 (;@6;)
                f32.const -inf (;=-inf;)
                local.set 16
                br 3 (;@3;)
              end
              local.get 8
              i32.const 0
              i32.lt_s
              br_if 1 (;@4;)
              local.get 16
              f32.const 0x1p+25 (;=33554432;)
              f32.mul
              i32.reinterpret_f32
              local.set 8
              i32.const -152
              local.set 2
            end
            local.get 8
            i32.const 4913933
            i32.add
            local.tee 8
            i32.const 23
            i32.shr_u
            local.get 2
            i32.add
            f32.convert_i32_s
            local.tee 17
            f32.const 0x1.62e3p-1 (;=0.6931381;)
            f32.mul
            local.get 8
            i32.const 8388607
            i32.and
            i32.const 1060439283
            i32.add
            f32.reinterpret_i32
            f32.const -0x1p+0 (;=-1;)
            f32.add
            local.tee 16
            local.get 17
            f32.const 0x1.2fefa2p-17 (;=0.000009058001;)
            f32.mul
            local.get 16
            local.get 16
            f32.const 0x1p+1 (;=2;)
            f32.add
            f32.div
            local.tee 17
            local.get 16
            local.get 16
            f32.const 0x1p-1 (;=0.5;)
            f32.mul
            f32.mul
            local.tee 18
            local.get 17
            local.get 17
            f32.mul
            local.tee 16
            local.get 16
            local.get 16
            f32.mul
            local.tee 16
            f32.const 0x1.23d3dcp-2 (;=0.28498787;)
            f32.mul
            f32.const 0x1.555554p-1 (;=0.6666666;)
            f32.add
            f32.mul
            local.get 16
            local.get 16
            f32.const 0x1.f13c4cp-3 (;=0.24279079;)
            f32.mul
            f32.const 0x1.999c26p-2 (;=0.40000972;)
            f32.add
            f32.mul
            f32.add
            f32.add
            f32.mul
            f32.add
            local.get 18
            f32.sub
            f32.add
            f32.add
            local.set 16
            br 1 (;@3;)
          end
          local.get 16
          local.get 16
          f32.sub
          f32.const 0x0p+0 (;=0;)
          f32.div
          local.set 16
        end
        local.get 5
        local.get 16
        f32.store
        local.get 6
        i32.const 4
        i32.add
        local.set 6
        local.get 5
        i32.const 4
        i32.add
        local.set 5
        local.get 3
        i32.const -1
        i32.add
        local.tee 3
        br_if 0 (;@2;)
      end
    end
  )
  (func (;25;) (type 12) (param i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
    (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 v128 f32 f32 f32 f32 f32 f32 f32 f32)
    global.get 0
    i32.const 16
    i32.sub
    local.tee 12
    global.set 0
    block ;; label = @1
      local.get 9
      i32.eqz
      br_if 0 (;@1;)
      local.get 10
      i32.eqz
      br_if 0 (;@1;)
      local.get 10
      i32.const 3
      i32.mul
      local.set 13
      local.get 10
      i32.const 1
      i32.shl
      local.set 14
      local.get 10
      i32.const 2
      i32.shl
      local.set 15
      local.get 11
      i32.const 2
      i32.shl
      local.set 16
      local.get 10
      local.get 10
      i32.mul
      local.tee 17
      i32.const 12
      i32.mul
      local.set 18
      local.get 17
      i32.const 3
      i32.shl
      local.set 19
      local.get 17
      i32.const 2
      i32.shl
      local.set 20
      local.get 11
      local.get 10
      i32.mul
      local.tee 17
      i32.const 12
      i32.mul
      local.set 21
      local.get 17
      i32.const 3
      i32.shl
      local.set 22
      local.get 17
      i32.const 2
      i32.shl
      local.set 23
      i32.const 0
      local.set 24
      loop ;; label = @2
        local.get 24
        local.get 10
        i32.mul
        local.set 25
        local.get 4
        local.set 26
        local.get 3
        local.set 27
        i32.const 0
        local.set 28
        loop ;; label = @3
          local.get 5
          local.get 28
          local.tee 29
          local.get 13
          i32.add
          i32.const 2
          i32.shl
          local.tee 17
          i32.add
          local.get 5
          local.get 29
          local.get 14
          i32.add
          i32.const 2
          i32.shl
          local.tee 30
          i32.add
          local.get 5
          local.get 29
          local.get 10
          i32.add
          i32.const 2
          i32.shl
          local.tee 31
          i32.add
          local.get 5
          local.get 29
          i32.const 2
          i32.shl
          local.tee 28
          i32.add
          v128.load32_zero
          v128.load32_lane 1
          v128.load32_lane 2
          v128.load32_lane 3
          local.get 6
          local.get 17
          i32.add
          local.get 6
          local.get 30
          i32.add
          local.get 6
          local.get 31
          i32.add
          local.get 6
          local.get 28
          i32.add
          v128.load32_zero
          v128.load32_lane 1
          v128.load32_lane 2
          v128.load32_lane 3
          f32x4.add
          local.set 32
          local.get 27
          local.set 17
          local.get 0
          local.set 30
          local.get 11
          local.set 31
          block ;; label = @4
            local.get 11
            i32.eqz
            br_if 0 (;@4;)
            loop ;; label = @5
              local.get 32
              local.get 30
              v128.load32_splat
              local.get 17
              local.get 21
              i32.add
              local.get 17
              local.get 22
              i32.add
              local.get 17
              local.get 23
              i32.add
              local.get 17
              v128.load32_zero
              v128.load32_lane 1
              v128.load32_lane 2
              v128.load32_lane 3
              f32x4.mul
              f32x4.add
              local.set 32
              local.get 17
              i32.const 4
              i32.add
              local.set 17
              local.get 30
              i32.const 4
              i32.add
              local.set 30
              local.get 31
              i32.const -1
              i32.add
              local.tee 31
              br_if 0 (;@5;)
            end
          end
          local.get 29
          i32.const 1
          i32.add
          local.set 28
          local.get 26
          local.set 17
          local.get 1
          local.set 30
          local.get 10
          local.set 31
          loop ;; label = @4
            local.get 32
            local.get 30
            v128.load32_splat
            local.get 17
            local.get 18
            i32.add
            local.get 17
            local.get 19
            i32.add
            local.get 17
            local.get 20
            i32.add
            local.get 17
            v128.load32_zero
            v128.load32_lane 1
            v128.load32_lane 2
            v128.load32_lane 3
            f32x4.mul
            f32x4.add
            local.set 32
            local.get 17
            i32.const 4
            i32.add
            local.set 17
            local.get 30
            i32.const 4
            i32.add
            local.set 30
            local.get 31
            i32.const -1
            i32.add
            local.tee 31
            br_if 0 (;@4;)
          end
          local.get 32
          f32x4.extract_lane 0
          local.tee 33
          f32.neg
          local.tee 34
          i32.reinterpret_f32
          local.tee 31
          i32.const 31
          i32.shr_u
          local.set 30
          block ;; label = @4
            block ;; label = @5
              block ;; label = @6
                block ;; label = @7
                  block ;; label = @8
                    block ;; label = @9
                      block ;; label = @10
                        local.get 33
                        i32.reinterpret_f32
                        i32.const 2147483647
                        i32.and
                        local.tee 17
                        i32.const 1118743632
                        i32.lt_u
                        br_if 0 (;@10;)
                        block ;; label = @11
                          local.get 17
                          i32.const 2139095040
                          i32.le_u
                          br_if 0 (;@11;)
                          local.get 34
                          local.set 35
                          br 7 (;@4;)
                        end
                        block ;; label = @11
                          local.get 17
                          i32.const 1118925335
                          i32.gt_u
                          br_if 0 (;@11;)
                          local.get 31
                          i32.const -1
                          i32.gt_s
                          br_if 2 (;@9;)
                          local.get 12
                          f32.const 0x1p-126 (;=0.000000000000000000000000000000000000011754944;)
                          local.get 33
                          f32.div
                          f32.store offset=12
                          local.get 12
                          f32.load offset=12
                          drop
                          br 2 (;@9;)
                        end
                        block ;; label = @11
                          local.get 31
                          i32.const -1
                          i32.gt_s
                          br_if 0 (;@11;)
                          local.get 12
                          f32.const 0x1p-126 (;=0.000000000000000000000000000000000000011754944;)
                          local.get 33
                          f32.div
                          f32.store offset=12
                          local.get 12
                          f32.load offset=12
                          drop
                          f32.const 0x0p+0 (;=0;)
                          local.set 35
                          local.get 17
                          i32.const 1120924084
                          i32.le_u
                          br_if 2 (;@9;)
                          br 7 (;@4;)
                        end
                        local.get 33
                        f32.const -0x1p+127 (;=-170141180000000000000000000000000000000;)
                        f32.mul
                        local.set 35
                        br 6 (;@4;)
                      end
                      block ;; label = @10
                        local.get 17
                        i32.const 1051816472
                        i32.gt_u
                        br_if 0 (;@10;)
                        local.get 17
                        i32.const 956301312
                        i32.le_u
                        br_if 2 (;@8;)
                        i32.const 0
                        local.set 17
                        f32.const 0x0p+0 (;=0;)
                        local.set 33
                        local.get 34
                        local.set 35
                        br 5 (;@5;)
                      end
                      local.get 17
                      i32.const 1065686418
                      i32.le_u
                      br_if 2 (;@7;)
                    end
                    local.get 33
                    f32.const -0x1.715476p+0 (;=-1.442695;)
                    f32.mul
                    local.get 30
                    i32.const 2
                    i32.shl
                    f32.load offset=1048840
                    f32.add
                    i32.trunc_sat_f32_s
                    local.set 17
                    br 2 (;@6;)
                  end
                  local.get 12
                  f32.const 0x1p+127 (;=170141180000000000000000000000000000000;)
                  local.get 33
                  f32.sub
                  f32.store offset=12
                  f32.const 0x1p+0 (;=1;)
                  local.get 33
                  f32.sub
                  local.set 35
                  local.get 12
                  f32.load offset=12
                  drop
                  br 3 (;@4;)
                end
                local.get 30
                i32.const 1
                i32.xor
                local.get 30
                i32.sub
                local.set 17
              end
              local.get 34
              local.get 17
              f32.convert_i32_s
              local.tee 35
              f32.const -0x1.62e4p-1 (;=-0.69314575;)
              f32.mul
              f32.add
              local.tee 34
              local.get 35
              f32.const 0x1.7f7d1cp-20 (;=0.0000014286068;)
              f32.mul
              local.tee 33
              f32.sub
              local.set 35
            end
            local.get 34
            local.get 35
            local.get 35
            local.get 35
            local.get 35
            f32.mul
            local.tee 36
            local.get 36
            f32.const -0x1.6aa42ap-9 (;=-0.0027667333;)
            f32.mul
            f32.const 0x1.55551ep-3 (;=0.16666625;)
            f32.add
            f32.mul
            f32.sub
            local.tee 36
            f32.mul
            f32.const 0x1p+1 (;=2;)
            local.get 36
            f32.sub
            f32.div
            local.get 33
            f32.sub
            f32.add
            f32.const 0x1p+0 (;=1;)
            f32.add
            local.set 35
            local.get 17
            i32.eqz
            br_if 0 (;@4;)
            block ;; label = @5
              block ;; label = @6
                block ;; label = @7
                  block ;; label = @8
                    local.get 17
                    i32.const 127
                    i32.gt_s
                    br_if 0 (;@8;)
                    local.get 17
                    i32.const -126
                    i32.ge_s
                    br_if 3 (;@5;)
                    local.get 35
                    f32.const 0x1p-102 (;=0.00000000000000000000000000000019721523;)
                    f32.mul
                    local.set 35
                    local.get 17
                    i32.const -229
                    i32.le_u
                    br_if 1 (;@7;)
                    local.get 17
                    i32.const 102
                    i32.add
                    local.set 17
                    br 3 (;@5;)
                  end
                  local.get 35
                  f32.const 0x1p+127 (;=170141180000000000000000000000000000000;)
                  f32.mul
                  local.set 35
                  local.get 17
                  i32.const 254
                  i32.gt_u
                  br_if 1 (;@6;)
                  local.get 17
                  i32.const -127
                  i32.add
                  local.set 17
                  br 2 (;@5;)
                end
                local.get 35
                f32.const 0x1p-102 (;=0.00000000000000000000000000000019721523;)
                f32.mul
                local.set 35
                local.get 17
                i32.const -330
                local.get 17
                i32.const -330
                i32.gt_u
                select
                i32.const 204
                i32.add
                local.set 17
                br 1 (;@5;)
              end
              local.get 35
              f32.const 0x1p+127 (;=170141180000000000000000000000000000000;)
              f32.mul
              local.set 35
              local.get 17
              i32.const 381
              local.get 17
              i32.const 381
              i32.lt_u
              select
              i32.const -254
              i32.add
              local.set 17
            end
            local.get 35
            local.get 17
            i32.const 23
            i32.shl
            i32.const 1065353216
            i32.add
            i32.const 2139095040
            i32.and
            f32.reinterpret_i32
            f32.mul
            local.set 35
          end
          local.get 32
          f32x4.extract_lane 1
          local.tee 34
          f32.neg
          local.tee 36
          i32.reinterpret_f32
          local.tee 31
          i32.const 31
          i32.shr_u
          local.set 30
          block ;; label = @4
            block ;; label = @5
              block ;; label = @6
                block ;; label = @7
                  block ;; label = @8
                    block ;; label = @9
                      block ;; label = @10
                        local.get 34
                        i32.reinterpret_f32
                        i32.const 2147483647
                        i32.and
                        local.tee 17
                        i32.const 1118743632
                        i32.lt_u
                        br_if 0 (;@10;)
                        block ;; label = @11
                          local.get 17
                          i32.const 2139095040
                          i32.le_u
                          br_if 0 (;@11;)
                          local.get 36
                          local.set 33
                          br 7 (;@4;)
                        end
                        block ;; label = @11
                          local.get 17
                          i32.const 1118925335
                          i32.gt_u
                          br_if 0 (;@11;)
                          local.get 31
                          i32.const -1
                          i32.gt_s
                          br_if 2 (;@9;)
                          local.get 12
                          f32.const 0x1p-126 (;=0.000000000000000000000000000000000000011754944;)
                          local.get 34
                          f32.div
                          f32.store offset=12
                          local.get 12
                          f32.load offset=12
                          drop
                          br 2 (;@9;)
                        end
                        block ;; label = @11
                          local.get 31
                          i32.const -1
                          i32.gt_s
                          br_if 0 (;@11;)
                          local.get 12
                          f32.const 0x1p-126 (;=0.000000000000000000000000000000000000011754944;)
                          local.get 34
                          f32.div
                          f32.store offset=12
                          local.get 12
                          f32.load offset=12
                          drop
                          f32.const 0x0p+0 (;=0;)
                          local.set 33
                          local.get 17
                          i32.const 1120924084
                          i32.le_u
                          br_if 2 (;@9;)
                          br 7 (;@4;)
                        end
                        local.get 34
                        f32.const -0x1p+127 (;=-170141180000000000000000000000000000000;)
                        f32.mul
                        local.set 33
                        br 6 (;@4;)
                      end
                      block ;; label = @10
                        local.get 17
                        i32.const 1051816472
                        i32.gt_u
                        br_if 0 (;@10;)
                        local.get 17
                        i32.const 956301312
                        i32.le_u
                        br_if 2 (;@8;)
                        i32.const 0
                        local.set 17
                        f32.const 0x0p+0 (;=0;)
                        local.set 34
                        local.get 36
                        local.set 33
                        br 5 (;@5;)
                      end
                      local.get 17
                      i32.const 1065686418
                      i32.le_u
                      br_if 2 (;@7;)
                    end
                    local.get 34
                    f32.const -0x1.715476p+0 (;=-1.442695;)
                    f32.mul
                    local.get 30
                    i32.const 2
                    i32.shl
                    f32.load offset=1048840
                    f32.add
                    i32.trunc_sat_f32_s
                    local.set 17
                    br 2 (;@6;)
                  end
                  local.get 12
                  f32.const 0x1p+127 (;=170141180000000000000000000000000000000;)
                  local.get 34
                  f32.sub
                  f32.store offset=12
                  f32.const 0x1p+0 (;=1;)
                  local.get 34
                  f32.sub
                  local.set 33
                  local.get 12
                  f32.load offset=12
                  drop
                  br 3 (;@4;)
                end
                local.get 30
                i32.const 1
                i32.xor
                local.get 30
                i32.sub
                local.set 17
              end
              local.get 36
              local.get 17
              f32.convert_i32_s
              local.tee 33
              f32.const -0x1.62e4p-1 (;=-0.69314575;)
              f32.mul
              f32.add
              local.tee 36
              local.get 33
              f32.const 0x1.7f7d1cp-20 (;=0.0000014286068;)
              f32.mul
              local.tee 34
              f32.sub
              local.set 33
            end
            local.get 36
            local.get 33
            local.get 33
            local.get 33
            local.get 33
            f32.mul
            local.tee 37
            local.get 37
            f32.const -0x1.6aa42ap-9 (;=-0.0027667333;)
            f32.mul
            f32.const 0x1.55551ep-3 (;=0.16666625;)
            f32.add
            f32.mul
            f32.sub
            local.tee 37
            f32.mul
            f32.const 0x1p+1 (;=2;)
            local.get 37
            f32.sub
            f32.div
            local.get 34
            f32.sub
            f32.add
            f32.const 0x1p+0 (;=1;)
            f32.add
            local.set 33
            local.get 17
            i32.eqz
            br_if 0 (;@4;)
            block ;; label = @5
              block ;; label = @6
                block ;; label = @7
                  block ;; label = @8
                    local.get 17
                    i32.const 127
                    i32.gt_s
                    br_if 0 (;@8;)
                    local.get 17
                    i32.const -126
                    i32.ge_s
                    br_if 3 (;@5;)
                    local.get 33
                    f32.const 0x1p-102 (;=0.00000000000000000000000000000019721523;)
                    f32.mul
                    local.set 33
                    local.get 17
                    i32.const -229
                    i32.le_u
                    br_if 1 (;@7;)
                    local.get 17
                    i32.const 102
                    i32.add
                    local.set 17
                    br 3 (;@5;)
                  end
                  local.get 33
                  f32.const 0x1p+127 (;=170141180000000000000000000000000000000;)
                  f32.mul
                  local.set 33
                  local.get 17
                  i32.const 254
                  i32.gt_u
                  br_if 1 (;@6;)
                  local.get 17
                  i32.const -127
                  i32.add
                  local.set 17
                  br 2 (;@5;)
                end
                local.get 33
                f32.const 0x1p-102 (;=0.00000000000000000000000000000019721523;)
                f32.mul
                local.set 33
                local.get 17
                i32.const -330
                local.get 17
                i32.const -330
                i32.gt_u
                select
                i32.const 204
                i32.add
                local.set 17
                br 1 (;@5;)
              end
              local.get 33
              f32.const 0x1p+127 (;=170141180000000000000000000000000000000;)
              f32.mul
              local.set 33
              local.get 17
              i32.const 381
              local.get 17
              i32.const 381
              i32.lt_u
              select
              i32.const -254
              i32.add
              local.set 17
            end
            local.get 33
            local.get 17
            i32.const 23
            i32.shl
            i32.const 1065353216
            i32.add
            i32.const 2139095040
            i32.and
            f32.reinterpret_i32
            f32.mul
            local.set 33
          end
          local.get 35
          f32.const 0x1p+0 (;=1;)
          f32.add
          local.set 34
          local.get 33
          f32.const 0x1p+0 (;=1;)
          f32.add
          local.set 36
          block ;; label = @4
            block ;; label = @5
              local.get 32
              f32x4.extract_lane 2
              local.tee 33
              f32.abs
              local.tee 35
              i32.reinterpret_f32
              local.tee 17
              i32.const 1057791828
              i32.gt_u
              br_if 0 (;@5;)
              block ;; label = @6
                block ;; label = @7
                  local.get 17
                  i32.const 1048757624
                  i32.gt_u
                  br_if 0 (;@7;)
                  local.get 17
                  i32.const 8388607
                  i32.gt_u
                  br_if 1 (;@6;)
                  local.get 12
                  local.get 33
                  local.get 33
                  f32.mul
                  f32.store offset=12
                  local.get 12
                  f32.load offset=12
                  drop
                  br 3 (;@4;)
                end
                block ;; label = @7
                  block ;; label = @8
                    block ;; label = @9
                      block ;; label = @10
                        block ;; label = @11
                          block ;; label = @12
                            block ;; label = @13
                              block ;; label = @14
                                block ;; label = @15
                                  local.get 35
                                  local.get 35
                                  f32.add
                                  local.tee 35
                                  i32.reinterpret_f32
                                  local.tee 30
                                  i32.const 2147483647
                                  i32.and
                                  local.tee 17
                                  i32.const 1100331075
                                  i32.gt_u
                                  br_if 0 (;@15;)
                                  local.get 17
                                  i32.const 1051816472
                                  i32.gt_u
                                  br_if 1 (;@14;)
                                  local.get 17
                                  i32.const 855638016
                                  i32.lt_u
                                  br_if 6 (;@9;)
                                  i32.const 0
                                  local.set 17
                                  f32.const 0x0p+0 (;=0;)
                                  local.set 38
                                  br 5 (;@10;)
                                end
                                local.get 35
                                f32.const -0x1p+0 (;=-1;)
                                local.get 17
                                i32.const 2139095040
                                i32.gt_u
                                local.tee 31
                                select
                                local.set 37
                                local.get 30
                                i32.const 0
                                i32.lt_s
                                br_if 7 (;@7;)
                                local.get 31
                                br_if 7 (;@7;)
                                f32.const 0x1p-1 (;=0.5;)
                                local.set 37
                                local.get 17
                                i32.const 1118925336
                                i32.lt_u
                                br_if 1 (;@13;)
                                local.get 35
                                f32.const 0x1p+127 (;=170141180000000000000000000000000000000;)
                                f32.mul
                                local.set 37
                                br 7 (;@7;)
                              end
                              local.get 17
                              i32.const 1065686418
                              i32.lt_u
                              br_if 1 (;@12;)
                              f32.const -0x1p-1 (;=-0.5;)
                              f32.const 0x1p-1 (;=0.5;)
                              local.get 30
                              i32.const 0
                              i32.lt_s
                              select
                              local.set 37
                            end
                            local.get 35
                            f32.const 0x1.715476p+0 (;=1.442695;)
                            f32.mul
                            local.get 37
                            f32.add
                            i32.trunc_sat_f32_s
                            local.tee 17
                            f32.convert_i32_s
                            local.tee 38
                            f32.const 0x1.2fefa2p-17 (;=0.000009058001;)
                            f32.mul
                            local.set 37
                            local.get 35
                            local.get 38
                            f32.const -0x1.62e3p-1 (;=-0.6931381;)
                            f32.mul
                            f32.add
                            local.set 38
                            br 1 (;@11;)
                          end
                          block ;; label = @12
                            local.get 30
                            i32.const 0
                            i32.lt_s
                            br_if 0 (;@12;)
                            local.get 35
                            f32.const -0x1.62e3p-1 (;=-0.6931381;)
                            f32.add
                            local.set 38
                            f32.const 0x1.2fefa2p-17 (;=0.000009058001;)
                            local.set 37
                            i32.const 1
                            local.set 17
                            br 1 (;@11;)
                          end
                          local.get 35
                          f32.const 0x1.62e3p-1 (;=0.6931381;)
                          f32.add
                          local.set 38
                          f32.const -0x1.2fefa2p-17 (;=-0.000009058001;)
                          local.set 37
                          i32.const -1
                          local.set 17
                        end
                        local.get 38
                        local.get 38
                        local.get 37
                        f32.sub
                        local.tee 35
                        f32.sub
                        local.get 37
                        f32.sub
                        local.set 38
                      end
                      local.get 35
                      local.get 35
                      f32.const 0x1p-1 (;=0.5;)
                      f32.mul
                      local.tee 39
                      f32.mul
                      local.tee 37
                      local.get 37
                      local.get 37
                      f32.const 0x1.9e602p-10 (;=0.001580717;)
                      f32.mul
                      f32.const -0x1.1110dp-5 (;=-0.033333212;)
                      f32.add
                      f32.mul
                      f32.const 0x1p+0 (;=1;)
                      f32.add
                      local.tee 40
                      f32.const 0x1.8p+1 (;=3;)
                      local.get 39
                      local.get 40
                      f32.mul
                      f32.sub
                      local.tee 39
                      f32.sub
                      f32.const 0x1.8p+2 (;=6;)
                      local.get 35
                      local.get 39
                      f32.mul
                      f32.sub
                      f32.div
                      f32.mul
                      local.set 39
                      local.get 17
                      br_if 1 (;@8;)
                      local.get 35
                      local.get 35
                      local.get 39
                      f32.mul
                      local.get 37
                      f32.sub
                      f32.sub
                      local.set 37
                      br 2 (;@7;)
                    end
                    block ;; label = @9
                      local.get 17
                      i32.const 8388608
                      i32.lt_u
                      br_if 0 (;@9;)
                      local.get 35
                      local.set 37
                      br 2 (;@7;)
                    end
                    local.get 12
                    local.get 35
                    local.get 35
                    f32.mul
                    f32.store offset=12
                    local.get 12
                    f32.load offset=12
                    drop
                    local.get 35
                    local.set 37
                    br 1 (;@7;)
                  end
                  local.get 35
                  local.get 39
                  local.get 38
                  f32.sub
                  f32.mul
                  local.get 38
                  f32.sub
                  local.get 37
                  f32.sub
                  local.set 37
                  block ;; label = @8
                    block ;; label = @9
                      block ;; label = @10
                        local.get 17
                        i32.const 1
                        i32.add
                        br_table 0 (;@10;) 2 (;@8;) 1 (;@9;) 2 (;@8;)
                      end
                      local.get 35
                      local.get 37
                      f32.sub
                      f32.const 0x1p-1 (;=0.5;)
                      f32.mul
                      f32.const -0x1p-1 (;=-0.5;)
                      f32.add
                      local.set 37
                      br 2 (;@7;)
                    end
                    block ;; label = @9
                      local.get 35
                      f32.const -0x1p-2 (;=-0.25;)
                      f32.lt
                      br_if 0 (;@9;)
                      local.get 35
                      local.get 37
                      f32.sub
                      local.tee 35
                      local.get 35
                      f32.add
                      f32.const 0x1p+0 (;=1;)
                      f32.add
                      local.set 37
                      br 2 (;@7;)
                    end
                    local.get 37
                    local.get 35
                    f32.const 0x1p-1 (;=0.5;)
                    f32.add
                    f32.sub
                    f32.const -0x1p+1 (;=-2;)
                    f32.mul
                    local.set 37
                    br 1 (;@7;)
                  end
                  local.get 17
                  i32.const 23
                  i32.shl
                  local.tee 30
                  i32.const 1065353216
                  i32.add
                  f32.reinterpret_i32
                  local.set 38
                  block ;; label = @8
                    local.get 17
                    i32.const 57
                    i32.lt_u
                    br_if 0 (;@8;)
                    local.get 35
                    local.get 37
                    f32.sub
                    f32.const 0x1p+0 (;=1;)
                    f32.add
                    local.tee 35
                    local.get 35
                    f32.add
                    f32.const 0x1p+127 (;=170141180000000000000000000000000000000;)
                    f32.mul
                    local.get 35
                    local.get 38
                    f32.mul
                    local.get 17
                    i32.const 128
                    i32.eq
                    select
                    f32.const -0x1p+0 (;=-1;)
                    f32.add
                    local.set 37
                    br 1 (;@7;)
                  end
                  i32.const 1065353216
                  local.get 30
                  i32.sub
                  f32.reinterpret_i32
                  local.set 39
                  block ;; label = @8
                    block ;; label = @9
                      local.get 17
                      i32.const 23
                      i32.lt_u
                      br_if 0 (;@9;)
                      local.get 35
                      local.get 37
                      local.get 39
                      f32.add
                      f32.sub
                      f32.const 0x1p+0 (;=1;)
                      f32.add
                      local.set 35
                      br 1 (;@8;)
                    end
                    f32.const 0x1p+0 (;=1;)
                    local.get 39
                    f32.sub
                    local.get 35
                    local.get 37
                    f32.sub
                    f32.add
                    local.set 35
                  end
                  local.get 35
                  local.get 38
                  f32.mul
                  local.set 37
                end
                local.get 37
                local.get 37
                f32.const 0x1p+1 (;=2;)
                f32.add
                f32.div
                local.set 35
                br 2 (;@4;)
              end
              local.get 35
              f32.const -0x1p+1 (;=-2;)
              f32.mul
              call 20
              local.tee 35
              f32.neg
              local.get 35
              f32.const 0x1p+1 (;=2;)
              f32.add
              f32.div
              local.set 35
              br 1 (;@4;)
            end
            block ;; label = @5
              local.get 17
              i32.const 1092616192
              i32.gt_u
              br_if 0 (;@5;)
              block ;; label = @6
                block ;; label = @7
                  block ;; label = @8
                    block ;; label = @9
                      block ;; label = @10
                        block ;; label = @11
                          block ;; label = @12
                            block ;; label = @13
                              block ;; label = @14
                                local.get 35
                                local.get 35
                                f32.add
                                local.tee 35
                                i32.reinterpret_f32
                                local.tee 30
                                i32.const 2147483647
                                i32.and
                                local.tee 17
                                i32.const 1100331075
                                i32.gt_u
                                br_if 0 (;@14;)
                                local.get 17
                                i32.const 1051816472
                                i32.gt_u
                                br_if 1 (;@13;)
                                local.get 17
                                i32.const 855638016
                                i32.lt_u
                                br_if 6 (;@8;)
                                i32.const 0
                                local.set 17
                                f32.const 0x0p+0 (;=0;)
                                local.set 38
                                br 5 (;@9;)
                              end
                              local.get 35
                              f32.const -0x1p+0 (;=-1;)
                              local.get 17
                              i32.const 2139095040
                              i32.gt_u
                              local.tee 31
                              select
                              local.set 37
                              local.get 30
                              i32.const 0
                              i32.lt_s
                              br_if 7 (;@6;)
                              local.get 31
                              br_if 7 (;@6;)
                              f32.const 0x1p-1 (;=0.5;)
                              local.set 37
                              local.get 17
                              i32.const 1118925336
                              i32.lt_u
                              br_if 1 (;@12;)
                              local.get 35
                              f32.const 0x1p+127 (;=170141180000000000000000000000000000000;)
                              f32.mul
                              local.set 37
                              br 7 (;@6;)
                            end
                            local.get 17
                            i32.const 1065686418
                            i32.lt_u
                            br_if 1 (;@11;)
                            f32.const -0x1p-1 (;=-0.5;)
                            f32.const 0x1p-1 (;=0.5;)
                            local.get 30
                            i32.const 0
                            i32.lt_s
                            select
                            local.set 37
                          end
                          local.get 35
                          f32.const 0x1.715476p+0 (;=1.442695;)
                          f32.mul
                          local.get 37
                          f32.add
                          i32.trunc_sat_f32_s
                          local.tee 17
                          f32.convert_i32_s
                          local.tee 38
                          f32.const 0x1.2fefa2p-17 (;=0.000009058001;)
                          f32.mul
                          local.set 37
                          local.get 35
                          local.get 38
                          f32.const -0x1.62e3p-1 (;=-0.6931381;)
                          f32.mul
                          f32.add
                          local.set 38
                          br 1 (;@10;)
                        end
                        block ;; label = @11
                          local.get 30
                          i32.const 0
                          i32.lt_s
                          br_if 0 (;@11;)
                          local.get 35
                          f32.const -0x1.62e3p-1 (;=-0.6931381;)
                          f32.add
                          local.set 38
                          f32.const 0x1.2fefa2p-17 (;=0.000009058001;)
                          local.set 37
                          i32.const 1
                          local.set 17
                          br 1 (;@10;)
                        end
                        local.get 35
                        f32.const 0x1.62e3p-1 (;=0.6931381;)
                        f32.add
                        local.set 38
                        f32.const -0x1.2fefa2p-17 (;=-0.000009058001;)
                        local.set 37
                        i32.const -1
                        local.set 17
                      end
                      local.get 38
                      local.get 38
                      local.get 37
                      f32.sub
                      local.tee 35
                      f32.sub
                      local.get 37
                      f32.sub
                      local.set 38
                    end
                    local.get 35
                    local.get 35
                    f32.const 0x1p-1 (;=0.5;)
                    f32.mul
                    local.tee 39
                    f32.mul
                    local.tee 37
                    local.get 37
                    local.get 37
                    f32.const 0x1.9e602p-10 (;=0.001580717;)
                    f32.mul
                    f32.const -0x1.1110dp-5 (;=-0.033333212;)
                    f32.add
                    f32.mul
                    f32.const 0x1p+0 (;=1;)
                    f32.add
                    local.tee 40
                    f32.const 0x1.8p+1 (;=3;)
                    local.get 39
                    local.get 40
                    f32.mul
                    f32.sub
                    local.tee 39
                    f32.sub
                    f32.const 0x1.8p+2 (;=6;)
                    local.get 35
                    local.get 39
                    f32.mul
                    f32.sub
                    f32.div
                    f32.mul
                    local.set 39
                    local.get 17
                    br_if 1 (;@7;)
                    local.get 35
                    local.get 35
                    local.get 39
                    f32.mul
                    local.get 37
                    f32.sub
                    f32.sub
                    local.set 37
                    br 2 (;@6;)
                  end
                  block ;; label = @8
                    local.get 17
                    i32.const 8388608
                    i32.lt_u
                    br_if 0 (;@8;)
                    local.get 35
                    local.set 37
                    br 2 (;@6;)
                  end
                  local.get 12
                  local.get 35
                  local.get 35
                  f32.mul
                  f32.store offset=12
                  local.get 12
                  f32.load offset=12
                  drop
                  local.get 35
                  local.set 37
                  br 1 (;@6;)
                end
                local.get 35
                local.get 39
                local.get 38
                f32.sub
                f32.mul
                local.get 38
                f32.sub
                local.get 37
                f32.sub
                local.set 37
                block ;; label = @7
                  block ;; label = @8
                    block ;; label = @9
                      local.get 17
                      i32.const 1
                      i32.add
                      br_table 0 (;@9;) 2 (;@7;) 1 (;@8;) 2 (;@7;)
                    end
                    local.get 35
                    local.get 37
                    f32.sub
                    f32.const 0x1p-1 (;=0.5;)
                    f32.mul
                    f32.const -0x1p-1 (;=-0.5;)
                    f32.add
                    local.set 37
                    br 2 (;@6;)
                  end
                  block ;; label = @8
                    local.get 35
                    f32.const -0x1p-2 (;=-0.25;)
                    f32.lt
                    br_if 0 (;@8;)
                    local.get 35
                    local.get 37
                    f32.sub
                    local.tee 35
                    local.get 35
                    f32.add
                    f32.const 0x1p+0 (;=1;)
                    f32.add
                    local.set 37
                    br 2 (;@6;)
                  end
                  local.get 37
                  local.get 35
                  f32.const 0x1p-1 (;=0.5;)
                  f32.add
                  f32.sub
                  f32.const -0x1p+1 (;=-2;)
                  f32.mul
                  local.set 37
                  br 1 (;@6;)
                end
                local.get 17
                i32.const 23
                i32.shl
                local.tee 30
                i32.const 1065353216
                i32.add
                f32.reinterpret_i32
                local.set 38
                block ;; label = @7
                  local.get 17
                  i32.const 57
                  i32.lt_u
                  br_if 0 (;@7;)
                  local.get 35
                  local.get 37
                  f32.sub
                  f32.const 0x1p+0 (;=1;)
                  f32.add
                  local.tee 35
                  local.get 35
                  f32.add
                  f32.const 0x1p+127 (;=170141180000000000000000000000000000000;)
                  f32.mul
                  local.get 35
                  local.get 38
                  f32.mul
                  local.get 17
                  i32.const 128
                  i32.eq
                  select
                  f32.const -0x1p+0 (;=-1;)
                  f32.add
                  local.set 37
                  br 1 (;@6;)
                end
                i32.const 1065353216
                local.get 30
                i32.sub
                f32.reinterpret_i32
                local.set 39
                block ;; label = @7
                  block ;; label = @8
                    local.get 17
                    i32.const 23
                    i32.lt_u
                    br_if 0 (;@8;)
                    local.get 35
                    local.get 37
                    local.get 39
                    f32.add
                    f32.sub
                    f32.const 0x1p+0 (;=1;)
                    f32.add
                    local.set 35
                    br 1 (;@7;)
                  end
                  f32.const 0x1p+0 (;=1;)
                  local.get 39
                  f32.sub
                  local.get 35
                  local.get 37
                  f32.sub
                  f32.add
                  local.set 35
                end
                local.get 35
                local.get 38
                f32.mul
                local.set 37
              end
              f32.const 0x1p+0 (;=1;)
              f32.const 0x1p+1 (;=2;)
              local.get 37
              f32.const 0x1p+1 (;=2;)
              f32.add
              f32.div
              f32.sub
              local.set 35
              br 1 (;@4;)
            end
            f32.const 0x0p+0 (;=0;)
            local.get 35
            f32.div
            f32.const 0x1p+0 (;=1;)
            f32.add
            local.set 35
          end
          f32.const 0x1p+0 (;=1;)
          local.get 34
          f32.div
          local.set 37
          f32.const 0x1p+0 (;=1;)
          local.get 36
          f32.div
          local.set 36
          local.get 35
          f32.neg
          local.get 35
          local.get 33
          i32.reinterpret_f32
          i32.const 0
          i32.lt_s
          select
          local.set 38
          local.get 32
          f32x4.extract_lane 3
          local.tee 33
          f32.neg
          local.tee 34
          i32.reinterpret_f32
          local.tee 31
          i32.const 31
          i32.shr_u
          local.set 30
          block ;; label = @4
            block ;; label = @5
              block ;; label = @6
                block ;; label = @7
                  block ;; label = @8
                    block ;; label = @9
                      block ;; label = @10
                        local.get 33
                        i32.reinterpret_f32
                        i32.const 2147483647
                        i32.and
                        local.tee 17
                        i32.const 1118743632
                        i32.lt_u
                        br_if 0 (;@10;)
                        block ;; label = @11
                          local.get 17
                          i32.const 2139095040
                          i32.le_u
                          br_if 0 (;@11;)
                          local.get 34
                          local.set 35
                          br 7 (;@4;)
                        end
                        block ;; label = @11
                          local.get 17
                          i32.const 1118925335
                          i32.gt_u
                          br_if 0 (;@11;)
                          local.get 31
                          i32.const -1
                          i32.gt_s
                          br_if 2 (;@9;)
                          local.get 12
                          f32.const 0x1p-126 (;=0.000000000000000000000000000000000000011754944;)
                          local.get 33
                          f32.div
                          f32.store offset=12
                          local.get 12
                          f32.load offset=12
                          drop
                          br 2 (;@9;)
                        end
                        block ;; label = @11
                          local.get 31
                          i32.const -1
                          i32.gt_s
                          br_if 0 (;@11;)
                          local.get 12
                          f32.const 0x1p-126 (;=0.000000000000000000000000000000000000011754944;)
                          local.get 33
                          f32.div
                          f32.store offset=12
                          local.get 12
                          f32.load offset=12
                          drop
                          f32.const 0x0p+0 (;=0;)
                          local.set 35
                          local.get 17
                          i32.const 1120924084
                          i32.le_u
                          br_if 2 (;@9;)
                          br 7 (;@4;)
                        end
                        local.get 33
                        f32.const -0x1p+127 (;=-170141180000000000000000000000000000000;)
                        f32.mul
                        local.set 35
                        br 6 (;@4;)
                      end
                      block ;; label = @10
                        local.get 17
                        i32.const 1051816472
                        i32.gt_u
                        br_if 0 (;@10;)
                        local.get 17
                        i32.const 956301312
                        i32.le_u
                        br_if 2 (;@8;)
                        i32.const 0
                        local.set 17
                        f32.const 0x0p+0 (;=0;)
                        local.set 33
                        local.get 34
                        local.set 35
                        br 5 (;@5;)
                      end
                      local.get 17
                      i32.const 1065686418
                      i32.le_u
                      br_if 2 (;@7;)
                    end
                    local.get 33
                    f32.const -0x1.715476p+0 (;=-1.442695;)
                    f32.mul
                    local.get 30
                    i32.const 2
                    i32.shl
                    f32.load offset=1048840
                    f32.add
                    i32.trunc_sat_f32_s
                    local.set 17
                    br 2 (;@6;)
                  end
                  local.get 12
                  f32.const 0x1p+127 (;=170141180000000000000000000000000000000;)
                  local.get 33
                  f32.sub
                  f32.store offset=12
                  f32.const 0x1p+0 (;=1;)
                  local.get 33
                  f32.sub
                  local.set 35
                  local.get 12
                  f32.load offset=12
                  drop
                  br 3 (;@4;)
                end
                local.get 30
                i32.const 1
                i32.xor
                local.get 30
                i32.sub
                local.set 17
              end
              local.get 34
              local.get 17
              f32.convert_i32_s
              local.tee 35
              f32.const -0x1.62e4p-1 (;=-0.69314575;)
              f32.mul
              f32.add
              local.tee 34
              local.get 35
              f32.const 0x1.7f7d1cp-20 (;=0.0000014286068;)
              f32.mul
              local.tee 33
              f32.sub
              local.set 35
            end
            local.get 34
            local.get 35
            local.get 35
            local.get 35
            local.get 35
            f32.mul
            local.tee 39
            local.get 39
            f32.const -0x1.6aa42ap-9 (;=-0.0027667333;)
            f32.mul
            f32.const 0x1.55551ep-3 (;=0.16666625;)
            f32.add
            f32.mul
            f32.sub
            local.tee 39
            f32.mul
            f32.const 0x1p+1 (;=2;)
            local.get 39
            f32.sub
            f32.div
            local.get 33
            f32.sub
            f32.add
            f32.const 0x1p+0 (;=1;)
            f32.add
            local.set 35
            local.get 17
            i32.eqz
            br_if 0 (;@4;)
            block ;; label = @5
              block ;; label = @6
                block ;; label = @7
                  block ;; label = @8
                    local.get 17
                    i32.const 127
                    i32.gt_s
                    br_if 0 (;@8;)
                    local.get 17
                    i32.const -126
                    i32.ge_s
                    br_if 3 (;@5;)
                    local.get 35
                    f32.const 0x1p-102 (;=0.00000000000000000000000000000019721523;)
                    f32.mul
                    local.set 35
                    local.get 17
                    i32.const -229
                    i32.le_u
                    br_if 1 (;@7;)
                    local.get 17
                    i32.const 102
                    i32.add
                    local.set 17
                    br 3 (;@5;)
                  end
                  local.get 35
                  f32.const 0x1p+127 (;=170141180000000000000000000000000000000;)
                  f32.mul
                  local.set 35
                  local.get 17
                  i32.const 254
                  i32.gt_u
                  br_if 1 (;@6;)
                  local.get 17
                  i32.const -127
                  i32.add
                  local.set 17
                  br 2 (;@5;)
                end
                local.get 35
                f32.const 0x1p-102 (;=0.00000000000000000000000000000019721523;)
                f32.mul
                local.set 35
                local.get 17
                i32.const -330
                local.get 17
                i32.const -330
                i32.gt_u
                select
                i32.const 204
                i32.add
                local.set 17
                br 1 (;@5;)
              end
              local.get 35
              f32.const 0x1p+127 (;=170141180000000000000000000000000000000;)
              f32.mul
              local.set 35
              local.get 17
              i32.const 381
              local.get 17
              i32.const 381
              i32.lt_u
              select
              i32.const -254
              i32.add
              local.set 17
            end
            local.get 35
            local.get 17
            i32.const 23
            i32.shl
            i32.const 1065353216
            i32.add
            i32.const 2139095040
            i32.and
            f32.reinterpret_i32
            f32.mul
            local.set 35
          end
          f32.const 0x1p+0 (;=1;)
          local.get 35
          f32.const 0x1p+0 (;=1;)
          f32.add
          f32.div
          local.set 34
          block ;; label = @4
            block ;; label = @5
              local.get 37
              local.get 38
              f32.mul
              local.get 36
              local.get 2
              local.get 29
              local.get 25
              i32.add
              i32.const 2
              i32.shl
              local.tee 17
              i32.add
              f32.load
              f32.mul
              f32.add
              local.tee 33
              f32.abs
              local.tee 35
              i32.reinterpret_f32
              local.tee 30
              i32.const 1057791828
              i32.gt_u
              br_if 0 (;@5;)
              block ;; label = @6
                block ;; label = @7
                  local.get 30
                  i32.const 1048757624
                  i32.gt_u
                  br_if 0 (;@7;)
                  local.get 30
                  i32.const 8388607
                  i32.gt_u
                  br_if 1 (;@6;)
                  local.get 12
                  local.get 33
                  local.get 33
                  f32.mul
                  f32.store offset=12
                  local.get 12
                  f32.load offset=12
                  drop
                  br 3 (;@4;)
                end
                block ;; label = @7
                  block ;; label = @8
                    block ;; label = @9
                      block ;; label = @10
                        block ;; label = @11
                          block ;; label = @12
                            block ;; label = @13
                              block ;; label = @14
                                block ;; label = @15
                                  local.get 35
                                  local.get 35
                                  f32.add
                                  local.tee 35
                                  i32.reinterpret_f32
                                  local.tee 31
                                  i32.const 2147483647
                                  i32.and
                                  local.tee 30
                                  i32.const 1100331075
                                  i32.gt_u
                                  br_if 0 (;@15;)
                                  local.get 30
                                  i32.const 1051816472
                                  i32.gt_u
                                  br_if 1 (;@14;)
                                  local.get 30
                                  i32.const 855638016
                                  i32.lt_u
                                  br_if 6 (;@9;)
                                  i32.const 0
                                  local.set 30
                                  f32.const 0x0p+0 (;=0;)
                                  local.set 37
                                  br 5 (;@10;)
                                end
                                local.get 35
                                f32.const -0x1p+0 (;=-1;)
                                local.get 30
                                i32.const 2139095040
                                i32.gt_u
                                local.tee 29
                                select
                                local.set 36
                                local.get 31
                                i32.const 0
                                i32.lt_s
                                br_if 7 (;@7;)
                                local.get 29
                                br_if 7 (;@7;)
                                f32.const 0x1p-1 (;=0.5;)
                                local.set 36
                                local.get 30
                                i32.const 1118925336
                                i32.lt_u
                                br_if 1 (;@13;)
                                local.get 35
                                f32.const 0x1p+127 (;=170141180000000000000000000000000000000;)
                                f32.mul
                                local.set 36
                                br 7 (;@7;)
                              end
                              local.get 30
                              i32.const 1065686418
                              i32.lt_u
                              br_if 1 (;@12;)
                              f32.const -0x1p-1 (;=-0.5;)
                              f32.const 0x1p-1 (;=0.5;)
                              local.get 31
                              i32.const 0
                              i32.lt_s
                              select
                              local.set 36
                            end
                            local.get 35
                            f32.const 0x1.715476p+0 (;=1.442695;)
                            f32.mul
                            local.get 36
                            f32.add
                            i32.trunc_sat_f32_s
                            local.tee 30
                            f32.convert_i32_s
                            local.tee 37
                            f32.const 0x1.2fefa2p-17 (;=0.000009058001;)
                            f32.mul
                            local.set 36
                            local.get 35
                            local.get 37
                            f32.const -0x1.62e3p-1 (;=-0.6931381;)
                            f32.mul
                            f32.add
                            local.set 37
                            br 1 (;@11;)
                          end
                          block ;; label = @12
                            local.get 31
                            i32.const 0
                            i32.lt_s
                            br_if 0 (;@12;)
                            local.get 35
                            f32.const -0x1.62e3p-1 (;=-0.6931381;)
                            f32.add
                            local.set 37
                            f32.const 0x1.2fefa2p-17 (;=0.000009058001;)
                            local.set 36
                            i32.const 1
                            local.set 30
                            br 1 (;@11;)
                          end
                          local.get 35
                          f32.const 0x1.62e3p-1 (;=0.6931381;)
                          f32.add
                          local.set 37
                          f32.const -0x1.2fefa2p-17 (;=-0.000009058001;)
                          local.set 36
                          i32.const -1
                          local.set 30
                        end
                        local.get 37
                        local.get 37
                        local.get 36
                        f32.sub
                        local.tee 35
                        f32.sub
                        local.get 36
                        f32.sub
                        local.set 37
                      end
                      local.get 35
                      local.get 35
                      f32.const 0x1p-1 (;=0.5;)
                      f32.mul
                      local.tee 38
                      f32.mul
                      local.tee 36
                      local.get 36
                      local.get 36
                      f32.const 0x1.9e602p-10 (;=0.001580717;)
                      f32.mul
                      f32.const -0x1.1110dp-5 (;=-0.033333212;)
                      f32.add
                      f32.mul
                      f32.const 0x1p+0 (;=1;)
                      f32.add
                      local.tee 39
                      f32.const 0x1.8p+1 (;=3;)
                      local.get 38
                      local.get 39
                      f32.mul
                      f32.sub
                      local.tee 38
                      f32.sub
                      f32.const 0x1.8p+2 (;=6;)
                      local.get 35
                      local.get 38
                      f32.mul
                      f32.sub
                      f32.div
                      f32.mul
                      local.set 38
                      local.get 30
                      br_if 1 (;@8;)
                      local.get 35
                      local.get 35
                      local.get 38
                      f32.mul
                      local.get 36
                      f32.sub
                      f32.sub
                      local.set 36
                      br 2 (;@7;)
                    end
                    block ;; label = @9
                      local.get 30
                      i32.const 8388608
                      i32.lt_u
                      br_if 0 (;@9;)
                      local.get 35
                      local.set 36
                      br 2 (;@7;)
                    end
                    local.get 12
                    local.get 35
                    local.get 35
                    f32.mul
                    f32.store offset=12
                    local.get 12
                    f32.load offset=12
                    drop
                    local.get 35
                    local.set 36
                    br 1 (;@7;)
                  end
                  local.get 35
                  local.get 38
                  local.get 37
                  f32.sub
                  f32.mul
                  local.get 37
                  f32.sub
                  local.get 36
                  f32.sub
                  local.set 36
                  block ;; label = @8
                    block ;; label = @9
                      block ;; label = @10
                        local.get 30
                        i32.const 1
                        i32.add
                        br_table 0 (;@10;) 2 (;@8;) 1 (;@9;) 2 (;@8;)
                      end
                      local.get 35
                      local.get 36
                      f32.sub
                      f32.const 0x1p-1 (;=0.5;)
                      f32.mul
                      f32.const -0x1p-1 (;=-0.5;)
                      f32.add
                      local.set 36
                      br 2 (;@7;)
                    end
                    block ;; label = @9
                      local.get 35
                      f32.const -0x1p-2 (;=-0.25;)
                      f32.lt
                      br_if 0 (;@9;)
                      local.get 35
                      local.get 36
                      f32.sub
                      local.tee 35
                      local.get 35
                      f32.add
                      f32.const 0x1p+0 (;=1;)
                      f32.add
                      local.set 36
                      br 2 (;@7;)
                    end
                    local.get 36
                    local.get 35
                    f32.const 0x1p-1 (;=0.5;)
                    f32.add
                    f32.sub
                    f32.const -0x1p+1 (;=-2;)
                    f32.mul
                    local.set 36
                    br 1 (;@7;)
                  end
                  local.get 30
                  i32.const 23
                  i32.shl
                  local.tee 31
                  i32.const 1065353216
                  i32.add
                  f32.reinterpret_i32
                  local.set 37
                  block ;; label = @8
                    local.get 30
                    i32.const 57
                    i32.lt_u
                    br_if 0 (;@8;)
                    local.get 35
                    local.get 36
                    f32.sub
                    f32.const 0x1p+0 (;=1;)
                    f32.add
                    local.tee 35
                    local.get 35
                    f32.add
                    f32.const 0x1p+127 (;=170141180000000000000000000000000000000;)
                    f32.mul
                    local.get 35
                    local.get 37
                    f32.mul
                    local.get 30
                    i32.const 128
                    i32.eq
                    select
                    f32.const -0x1p+0 (;=-1;)
                    f32.add
                    local.set 36
                    br 1 (;@7;)
                  end
                  i32.const 1065353216
                  local.get 31
                  i32.sub
                  f32.reinterpret_i32
                  local.set 38
                  block ;; label = @8
                    block ;; label = @9
                      local.get 30
                      i32.const 23
                      i32.lt_u
                      br_if 0 (;@9;)
                      local.get 35
                      local.get 36
                      local.get 38
                      f32.add
                      f32.sub
                      f32.const 0x1p+0 (;=1;)
                      f32.add
                      local.set 35
                      br 1 (;@8;)
                    end
                    f32.const 0x1p+0 (;=1;)
                    local.get 38
                    f32.sub
                    local.get 35
                    local.get 36
                    f32.sub
                    f32.add
                    local.set 35
                  end
                  local.get 35
                  local.get 37
                  f32.mul
                  local.set 36
                end
                local.get 36
                local.get 36
                f32.const 0x1p+1 (;=2;)
                f32.add
                f32.div
                local.set 35
                br 2 (;@4;)
              end
              local.get 35
              f32.const -0x1p+1 (;=-2;)
              f32.mul
              call 20
              local.tee 35
              f32.neg
              local.get 35
              f32.const 0x1p+1 (;=2;)
              f32.add
              f32.div
              local.set 35
              br 1 (;@4;)
            end
            block ;; label = @5
              local.get 30
              i32.const 1092616192
              i32.gt_u
              br_if 0 (;@5;)
              block ;; label = @6
                block ;; label = @7
                  block ;; label = @8
                    block ;; label = @9
                      block ;; label = @10
                        block ;; label = @11
                          block ;; label = @12
                            block ;; label = @13
                              block ;; label = @14
                                local.get 35
                                local.get 35
                                f32.add
                                local.tee 35
                                i32.reinterpret_f32
                                local.tee 31
                                i32.const 2147483647
                                i32.and
                                local.tee 30
                                i32.const 1100331075
                                i32.gt_u
                                br_if 0 (;@14;)
                                local.get 30
                                i32.const 1051816472
                                i32.gt_u
                                br_if 1 (;@13;)
                                local.get 30
                                i32.const 855638016
                                i32.lt_u
                                br_if 6 (;@8;)
                                i32.const 0
                                local.set 30
                                f32.const 0x0p+0 (;=0;)
                                local.set 37
                                br 5 (;@9;)
                              end
                              local.get 35
                              f32.const -0x1p+0 (;=-1;)
                              local.get 30
                              i32.const 2139095040
                              i32.gt_u
                              local.tee 29
                              select
                              local.set 36
                              local.get 31
                              i32.const 0
                              i32.lt_s
                              br_if 7 (;@6;)
                              local.get 29
                              br_if 7 (;@6;)
                              f32.const 0x1p-1 (;=0.5;)
                              local.set 36
                              local.get 30
                              i32.const 1118925336
                              i32.lt_u
                              br_if 1 (;@12;)
                              local.get 35
                              f32.const 0x1p+127 (;=170141180000000000000000000000000000000;)
                              f32.mul
                              local.set 36
                              br 7 (;@6;)
                            end
                            local.get 30
                            i32.const 1065686418
                            i32.lt_u
                            br_if 1 (;@11;)
                            f32.const -0x1p-1 (;=-0.5;)
                            f32.const 0x1p-1 (;=0.5;)
                            local.get 31
                            i32.const 0
                            i32.lt_s
                            select
                            local.set 36
                          end
                          local.get 35
                          f32.const 0x1.715476p+0 (;=1.442695;)
                          f32.mul
                          local.get 36
                          f32.add
                          i32.trunc_sat_f32_s
                          local.tee 30
                          f32.convert_i32_s
                          local.tee 37
                          f32.const 0x1.2fefa2p-17 (;=0.000009058001;)
                          f32.mul
                          local.set 36
                          local.get 35
                          local.get 37
                          f32.const -0x1.62e3p-1 (;=-0.6931381;)
                          f32.mul
                          f32.add
                          local.set 37
                          br 1 (;@10;)
                        end
                        block ;; label = @11
                          local.get 31
                          i32.const 0
                          i32.lt_s
                          br_if 0 (;@11;)
                          local.get 35
                          f32.const -0x1.62e3p-1 (;=-0.6931381;)
                          f32.add
                          local.set 37
                          f32.const 0x1.2fefa2p-17 (;=0.000009058001;)
                          local.set 36
                          i32.const 1
                          local.set 30
                          br 1 (;@10;)
                        end
                        local.get 35
                        f32.const 0x1.62e3p-1 (;=0.6931381;)
                        f32.add
                        local.set 37
                        f32.const -0x1.2fefa2p-17 (;=-0.000009058001;)
                        local.set 36
                        i32.const -1
                        local.set 30
                      end
                      local.get 37
                      local.get 37
                      local.get 36
                      f32.sub
                      local.tee 35
                      f32.sub
                      local.get 36
                      f32.sub
                      local.set 37
                    end
                    local.get 35
                    local.get 35
                    f32.const 0x1p-1 (;=0.5;)
                    f32.mul
                    local.tee 38
                    f32.mul
                    local.tee 36
                    local.get 36
                    local.get 36
                    f32.const 0x1.9e602p-10 (;=0.001580717;)
                    f32.mul
                    f32.const -0x1.1110dp-5 (;=-0.033333212;)
                    f32.add
                    f32.mul
                    f32.const 0x1p+0 (;=1;)
                    f32.add
                    local.tee 39
                    f32.const 0x1.8p+1 (;=3;)
                    local.get 38
                    local.get 39
                    f32.mul
                    f32.sub
                    local.tee 38
                    f32.sub
                    f32.const 0x1.8p+2 (;=6;)
                    local.get 35
                    local.get 38
                    f32.mul
                    f32.sub
                    f32.div
                    f32.mul
                    local.set 38
                    local.get 30
                    br_if 1 (;@7;)
                    local.get 35
                    local.get 35
                    local.get 38
                    f32.mul
                    local.get 36
                    f32.sub
                    f32.sub
                    local.set 36
                    br 2 (;@6;)
                  end
                  block ;; label = @8
                    local.get 30
                    i32.const 8388608
                    i32.lt_u
                    br_if 0 (;@8;)
                    local.get 35
                    local.set 36
                    br 2 (;@6;)
                  end
                  local.get 12
                  local.get 35
                  local.get 35
                  f32.mul
                  f32.store offset=12
                  local.get 12
                  f32.load offset=12
                  drop
                  local.get 35
                  local.set 36
                  br 1 (;@6;)
                end
                local.get 35
                local.get 38
                local.get 37
                f32.sub
                f32.mul
                local.get 37
                f32.sub
                local.get 36
                f32.sub
                local.set 36
                block ;; label = @7
                  block ;; label = @8
                    block ;; label = @9
                      local.get 30
                      i32.const 1
                      i32.add
                      br_table 0 (;@9;) 2 (;@7;) 1 (;@8;) 2 (;@7;)
                    end
                    local.get 35
                    local.get 36
                    f32.sub
                    f32.const 0x1p-1 (;=0.5;)
                    f32.mul
                    f32.const -0x1p-1 (;=-0.5;)
                    f32.add
                    local.set 36
                    br 2 (;@6;)
                  end
                  block ;; label = @8
                    local.get 35
                    f32.const -0x1p-2 (;=-0.25;)
                    f32.lt
                    br_if 0 (;@8;)
                    local.get 35
                    local.get 36
                    f32.sub
                    local.tee 35
                    local.get 35
                    f32.add
                    f32.const 0x1p+0 (;=1;)
                    f32.add
                    local.set 36
                    br 2 (;@6;)
                  end
                  local.get 36
                  local.get 35
                  f32.const 0x1p-1 (;=0.5;)
                  f32.add
                  f32.sub
                  f32.const -0x1p+1 (;=-2;)
                  f32.mul
                  local.set 36
                  br 1 (;@6;)
                end
                local.get 30
                i32.const 23
                i32.shl
                local.tee 31
                i32.const 1065353216
                i32.add
                f32.reinterpret_i32
                local.set 37
                block ;; label = @7
                  local.get 30
                  i32.const 57
                  i32.lt_u
                  br_if 0 (;@7;)
                  local.get 35
                  local.get 36
                  f32.sub
                  f32.const 0x1p+0 (;=1;)
                  f32.add
                  local.tee 35
                  local.get 35
                  f32.add
                  f32.const 0x1p+127 (;=170141180000000000000000000000000000000;)
                  f32.mul
                  local.get 35
                  local.get 37
                  f32.mul
                  local.get 30
                  i32.const 128
                  i32.eq
                  select
                  f32.const -0x1p+0 (;=-1;)
                  f32.add
                  local.set 36
                  br 1 (;@6;)
                end
                i32.const 1065353216
                local.get 31
                i32.sub
                f32.reinterpret_i32
                local.set 38
                block ;; label = @7
                  block ;; label = @8
                    local.get 30
                    i32.const 23
                    i32.lt_u
                    br_if 0 (;@8;)
                    local.get 35
                    local.get 36
                    local.get 38
                    f32.add
                    f32.sub
                    f32.const 0x1p+0 (;=1;)
                    f32.add
                    local.set 35
                    br 1 (;@7;)
                  end
                  f32.const 0x1p+0 (;=1;)
                  local.get 38
                  f32.sub
                  local.get 35
                  local.get 36
                  f32.sub
                  f32.add
                  local.set 35
                end
                local.get 35
                local.get 37
                f32.mul
                local.set 36
              end
              f32.const 0x1p+0 (;=1;)
              f32.const 0x1p+1 (;=2;)
              local.get 36
              f32.const 0x1p+1 (;=2;)
              f32.add
              f32.div
              f32.sub
              local.set 35
              br 1 (;@4;)
            end
            f32.const 0x0p+0 (;=0;)
            local.get 35
            f32.div
            f32.const 0x1p+0 (;=1;)
            f32.add
            local.set 35
          end
          local.get 7
          local.get 17
          i32.add
          local.get 34
          local.get 35
          f32.neg
          local.get 35
          local.get 33
          i32.reinterpret_f32
          i32.const 0
          i32.lt_s
          select
          f32.mul
          f32.store
          local.get 8
          local.get 17
          i32.add
          local.get 33
          f32.store
          local.get 26
          local.get 15
          i32.add
          local.set 26
          local.get 27
          local.get 16
          i32.add
          local.set 27
          local.get 28
          local.get 10
          i32.ne
          br_if 0 (;@3;)
        end
        local.get 1
        local.get 15
        i32.add
        local.set 1
        local.get 0
        local.get 16
        i32.add
        local.set 0
        local.get 24
        i32.const 1
        i32.add
        local.tee 24
        local.get 9
        i32.ne
        br_if 0 (;@2;)
      end
    end
    local.get 12
    i32.const 16
    i32.add
    global.set 0
  )
  (func (;26;) (type 13) (param i32 i32 i32 i32 i32 i32 i32)
    (local i32 i32 i32 i32 i32 i32 i32 i32 f32 v128)
    block ;; label = @1
      block ;; label = @2
        local.get 3
        local.get 2
        i32.le_u
        br_if 0 (;@2;)
        block ;; label = @3
          local.get 4
          i32.eqz
          br_if 0 (;@3;)
          local.get 6
          local.get 4
          i32.const 2
          i32.shl
          i32.const -4
          i32.add
          local.tee 7
          i32.add
          local.set 8
          local.get 5
          local.get 7
          i32.add
          local.set 9
          loop ;; label = @4
            local.get 2
            i32.const 1
            i32.add
            local.set 10
            i32.const 0
            local.set 11
            local.get 8
            local.set 5
            local.get 9
            local.set 6
            local.get 4
            local.set 12
            local.get 2
            local.set 7
            loop ;; label = @5
              local.get 6
              i32.load
              local.tee 13
              i32.eqz
              br_if 4 (;@1;)
              local.get 6
              i32.const -4
              i32.add
              local.set 6
              local.get 5
              i32.load
              local.get 7
              local.get 7
              local.get 13
              i32.div_u
              local.tee 14
              local.get 13
              i32.mul
              i32.sub
              i32.mul
              local.get 11
              i32.add
              local.set 11
              local.get 5
              i32.const -4
              i32.add
              local.set 5
              local.get 14
              local.set 7
              local.get 12
              i32.const -1
              i32.add
              local.tee 12
              br_if 0 (;@5;)
            end
            local.get 1
            local.get 2
            i32.const 2
            i32.shl
            i32.add
            local.get 0
            local.get 11
            i32.const 2
            i32.shl
            i32.add
            f32.load
            f32.store
            local.get 10
            local.set 2
            local.get 10
            local.get 3
            i32.ne
            br_if 0 (;@4;)
            br 2 (;@2;)
          end
        end
        local.get 0
        f32.load
        local.set 15
        block ;; label = @3
          local.get 3
          local.get 2
          i32.sub
          local.tee 7
          i32.const 4
          i32.lt_u
          br_if 0 (;@3;)
          local.get 1
          local.get 2
          i32.const 2
          i32.shl
          i32.add
          local.set 6
          local.get 2
          local.get 7
          i32.const -4
          i32.and
          local.tee 13
          i32.add
          local.set 2
          local.get 15
          f32x4.splat
          local.set 16
          local.get 13
          local.set 5
          loop ;; label = @4
            local.get 6
            local.get 16
            v128.store align=4
            local.get 6
            i32.const 16
            i32.add
            local.set 6
            local.get 5
            i32.const -4
            i32.add
            local.tee 5
            br_if 0 (;@4;)
          end
          local.get 7
          local.get 13
          i32.eq
          br_if 1 (;@2;)
        end
        local.get 3
        local.get 2
        i32.sub
        local.set 5
        local.get 1
        local.get 2
        i32.const 2
        i32.shl
        i32.add
        local.set 6
        loop ;; label = @3
          local.get 6
          local.get 15
          f32.store
          local.get 6
          i32.const 4
          i32.add
          local.set 6
          local.get 5
          i32.const -1
          i32.add
          local.tee 5
          br_if 0 (;@3;)
        end
      end
      return
    end
    call 3
    unreachable
  )
  (func (;27;) (type 1) (param i32 i32 i32 i32 i32)
    (local i32 i32 i32 i32 i32 i32)
    block ;; label = @1
      local.get 3
      i32.const 4
      i32.add
      local.get 4
      i32.gt_u
      br_if 0 (;@1;)
      local.get 2
      local.get 3
      i32.const 2
      i32.shl
      local.tee 5
      i32.add
      local.set 6
      local.get 1
      local.get 5
      i32.add
      local.set 7
      local.get 0
      local.get 5
      i32.add
      local.set 5
      loop ;; label = @2
        local.get 6
        local.get 5
        v128.load align=1
        local.get 7
        v128.load align=1
        f32x4.mul
        v128.store align=1
        local.get 6
        i32.const 16
        i32.add
        local.set 6
        local.get 7
        i32.const 16
        i32.add
        local.set 7
        local.get 5
        i32.const 16
        i32.add
        local.set 5
        local.get 3
        local.tee 8
        i32.const 4
        i32.add
        local.set 3
        local.get 8
        i32.const 8
        i32.add
        local.get 4
        i32.le_u
        br_if 0 (;@2;)
      end
    end
    block ;; label = @1
      local.get 4
      local.get 3
      i32.le_u
      br_if 0 (;@1;)
      block ;; label = @2
        local.get 4
        local.get 3
        i32.sub
        local.tee 9
        i32.const 8
        i32.lt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 2
        i32.sub
        i32.const -16
        i32.gt_u
        br_if 0 (;@2;)
        local.get 1
        local.get 2
        i32.sub
        i32.const -16
        i32.gt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 3
        i32.const 2
        i32.shl
        local.tee 5
        i32.add
        local.set 6
        local.get 1
        local.get 5
        i32.add
        local.set 7
        local.get 2
        local.get 5
        i32.add
        local.set 5
        local.get 3
        local.get 9
        i32.const -4
        i32.and
        local.tee 10
        i32.add
        local.set 3
        local.get 10
        local.set 8
        loop ;; label = @3
          local.get 5
          local.get 6
          v128.load align=4
          local.get 7
          v128.load align=4
          f32x4.mul
          v128.store align=4
          local.get 6
          i32.const 16
          i32.add
          local.set 6
          local.get 7
          i32.const 16
          i32.add
          local.set 7
          local.get 5
          i32.const 16
          i32.add
          local.set 5
          local.get 8
          i32.const -4
          i32.add
          local.tee 8
          br_if 0 (;@3;)
        end
        local.get 9
        local.get 10
        i32.eq
        br_if 1 (;@1;)
      end
      local.get 3
      i32.const 1
      i32.add
      local.set 6
      block ;; label = @2
        local.get 4
        local.get 3
        i32.sub
        i32.const 1
        i32.and
        i32.eqz
        br_if 0 (;@2;)
        local.get 2
        local.get 3
        i32.const 2
        i32.shl
        local.tee 7
        i32.add
        local.get 0
        local.get 7
        i32.add
        f32.load
        local.get 1
        local.get 7
        i32.add
        f32.load
        f32.mul
        f32.store
        local.get 6
        local.set 3
      end
      local.get 4
      local.get 6
      i32.eq
      br_if 0 (;@1;)
      local.get 4
      local.get 3
      i32.sub
      local.set 8
      local.get 0
      local.get 3
      i32.const 2
      i32.shl
      local.tee 5
      i32.add
      local.set 6
      local.get 1
      local.get 5
      i32.add
      local.set 7
      local.get 2
      local.get 5
      i32.add
      local.set 5
      loop ;; label = @2
        local.get 5
        local.get 6
        f32.load
        local.get 7
        f32.load
        f32.mul
        f32.store
        local.get 5
        i32.const 4
        i32.add
        local.get 6
        i32.const 4
        i32.add
        f32.load
        local.get 7
        i32.const 4
        i32.add
        f32.load
        f32.mul
        f32.store
        local.get 6
        i32.const 8
        i32.add
        local.set 6
        local.get 7
        i32.const 8
        i32.add
        local.set 7
        local.get 5
        i32.const 8
        i32.add
        local.set 5
        local.get 8
        i32.const -2
        i32.add
        local.tee 8
        br_if 0 (;@2;)
      end
    end
  )
  (func (;28;) (type 6) (param i32 i32 i32 i32)
    (local i32 i32 i32 i32 i32)
    block ;; label = @1
      local.get 2
      i32.const 4
      i32.add
      local.get 3
      i32.gt_u
      br_if 0 (;@1;)
      local.get 1
      local.get 2
      i32.const 2
      i32.shl
      local.tee 4
      i32.add
      local.set 5
      local.get 0
      local.get 4
      i32.add
      local.set 4
      loop ;; label = @2
        local.get 5
        v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
        local.get 4
        v128.load align=1
        f32x4.sub
        v128.store align=1
        local.get 5
        i32.const 16
        i32.add
        local.set 5
        local.get 4
        i32.const 16
        i32.add
        local.set 4
        local.get 2
        local.tee 6
        i32.const 4
        i32.add
        local.set 2
        local.get 6
        i32.const 8
        i32.add
        local.get 3
        i32.le_u
        br_if 0 (;@2;)
      end
    end
    block ;; label = @1
      local.get 3
      local.get 2
      i32.le_u
      br_if 0 (;@1;)
      block ;; label = @2
        local.get 3
        local.get 2
        i32.sub
        local.tee 7
        i32.const 4
        i32.lt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 1
        i32.sub
        i32.const -16
        i32.gt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 2
        i32.const 2
        i32.shl
        local.tee 4
        i32.add
        local.set 5
        local.get 1
        local.get 4
        i32.add
        local.set 4
        local.get 2
        local.get 7
        i32.const -4
        i32.and
        local.tee 8
        i32.add
        local.set 2
        local.get 8
        local.set 6
        loop ;; label = @3
          local.get 4
          local.get 5
          v128.load align=4
          f32x4.neg
          v128.store align=4
          local.get 5
          i32.const 16
          i32.add
          local.set 5
          local.get 4
          i32.const 16
          i32.add
          local.set 4
          local.get 6
          i32.const -4
          i32.add
          local.tee 6
          br_if 0 (;@3;)
        end
        local.get 7
        local.get 8
        i32.eq
        br_if 1 (;@1;)
      end
      local.get 2
      local.set 7
      block ;; label = @2
        local.get 3
        local.get 2
        i32.sub
        i32.const 3
        i32.and
        local.tee 6
        i32.eqz
        br_if 0 (;@2;)
        local.get 2
        local.get 6
        i32.add
        local.set 7
        local.get 0
        local.get 2
        i32.const 2
        i32.shl
        local.tee 4
        i32.add
        local.set 5
        local.get 1
        local.get 4
        i32.add
        local.set 4
        loop ;; label = @3
          local.get 4
          local.get 5
          f32.load
          f32.neg
          f32.store
          local.get 5
          i32.const 4
          i32.add
          local.set 5
          local.get 4
          i32.const 4
          i32.add
          local.set 4
          local.get 6
          i32.const -1
          i32.add
          local.tee 6
          br_if 0 (;@3;)
        end
      end
      local.get 2
      local.get 3
      i32.sub
      i32.const -4
      i32.gt_u
      br_if 0 (;@1;)
      local.get 7
      i32.const 2
      i32.shl
      local.set 2
      local.get 3
      local.get 7
      i32.sub
      local.set 6
      loop ;; label = @2
        local.get 1
        local.get 2
        i32.add
        local.tee 5
        local.get 0
        local.get 2
        i32.add
        local.tee 4
        f32.load
        f32.neg
        f32.store
        local.get 5
        i32.const 4
        i32.add
        local.get 4
        i32.const 4
        i32.add
        f32.load
        f32.neg
        f32.store
        local.get 5
        i32.const 8
        i32.add
        local.get 4
        i32.const 8
        i32.add
        f32.load
        f32.neg
        f32.store
        local.get 5
        i32.const 12
        i32.add
        local.get 4
        i32.const 12
        i32.add
        f32.load
        f32.neg
        f32.store
        local.get 0
        i32.const 16
        i32.add
        local.set 0
        local.get 1
        i32.const 16
        i32.add
        local.set 1
        local.get 6
        i32.const -4
        i32.add
        local.tee 6
        br_if 0 (;@2;)
      end
    end
  )
  (func (;29;) (type 6) (param i32 i32 i32 i32)
    (local i32 f32 f32 f32 f32)
    block ;; label = @1
      local.get 2
      local.get 1
      i32.le_u
      br_if 0 (;@1;)
      local.get 2
      local.get 1
      i32.sub
      local.set 4
      local.get 3
      i32.const -1640531527
      local.get 3
      select
      local.set 2
      local.get 0
      local.get 1
      i32.const 2
      i32.shl
      i32.add
      local.set 1
      loop ;; label = @2
        local.get 2
        i32.const 13
        i32.shl
        local.get 2
        i32.xor
        local.tee 2
        i32.const 17
        i32.shr_u
        local.get 2
        i32.xor
        local.tee 2
        i32.const 5
        i32.shl
        local.get 2
        i32.xor
        local.tee 3
        i32.const 13
        i32.shl
        local.get 3
        i32.xor
        local.tee 2
        i32.const 17
        i32.shr_u
        local.get 2
        i32.xor
        local.tee 2
        i32.const 5
        i32.shl
        local.get 2
        i32.xor
        local.tee 2
        f32.convert_i32_u
        f32.const 0x1p+0 (;=1;)
        f32.add
        f32.const 0x1p-32 (;=0.00000000023283064;)
        f32.mul
        local.set 5
        block ;; label = @3
          block ;; label = @4
            block ;; label = @5
              local.get 3
              f32.convert_i32_u
              f32.const 0x1p+0 (;=1;)
              f32.add
              f32.const 0x1p-32 (;=0.00000000023283064;)
              f32.mul
              local.tee 6
              i32.reinterpret_f32
              local.tee 3
              i32.const 8388608
              i32.lt_u
              br_if 0 (;@5;)
              local.get 3
              i32.const 2139095039
              i32.gt_u
              br_if 2 (;@3;)
              i32.const -127
              local.set 0
              local.get 3
              i32.const 1065353216
              i32.ne
              br_if 1 (;@4;)
              f32.const 0x0p+0 (;=0;)
              local.set 6
              br 2 (;@3;)
            end
            block ;; label = @5
              local.get 6
              f32.const 0x0p+0 (;=0;)
              f32.ne
              br_if 0 (;@5;)
              f32.const -inf (;=-inf;)
              local.set 6
              br 2 (;@3;)
            end
            local.get 6
            f32.const 0x1p+25 (;=33554432;)
            f32.mul
            i32.reinterpret_f32
            local.set 3
            i32.const -152
            local.set 0
          end
          local.get 3
          i32.const 4913933
          i32.add
          local.tee 3
          i32.const 23
          i32.shr_u
          local.get 0
          i32.add
          f32.convert_i32_s
          local.tee 7
          f32.const 0x1.62e3p-1 (;=0.6931381;)
          f32.mul
          local.get 3
          i32.const 8388607
          i32.and
          i32.const 1060439283
          i32.add
          f32.reinterpret_i32
          f32.const -0x1p+0 (;=-1;)
          f32.add
          local.tee 6
          local.get 7
          f32.const 0x1.2fefa2p-17 (;=0.000009058001;)
          f32.mul
          local.get 6
          local.get 6
          f32.const 0x1p+1 (;=2;)
          f32.add
          f32.div
          local.tee 7
          local.get 6
          local.get 6
          f32.const 0x1p-1 (;=0.5;)
          f32.mul
          f32.mul
          local.tee 8
          local.get 7
          local.get 7
          f32.mul
          local.tee 6
          local.get 6
          local.get 6
          f32.mul
          local.tee 6
          f32.const 0x1.23d3dcp-2 (;=0.28498787;)
          f32.mul
          f32.const 0x1.555554p-1 (;=0.6666666;)
          f32.add
          f32.mul
          local.get 6
          local.get 6
          f32.const 0x1.f13c4cp-3 (;=0.24279079;)
          f32.mul
          f32.const 0x1.999c26p-2 (;=0.40000972;)
          f32.add
          f32.mul
          f32.add
          f32.add
          f32.mul
          f32.add
          local.get 8
          f32.sub
          f32.add
          f32.add
          local.set 6
        end
        local.get 1
        local.get 6
        f32.const -0x1p+1 (;=-2;)
        f32.mul
        call 30
        local.get 5
        f32.const 0x1.921fb6p+2 (;=6.2831855;)
        f32.mul
        call 12
        f32.mul
        f32.store
        local.get 1
        i32.const 4
        i32.add
        local.set 1
        local.get 4
        i32.const -1
        i32.add
        local.tee 4
        br_if 0 (;@2;)
      end
    end
  )
  (func (;30;) (type 7) (param f32) (result f32)
    (local i32 f32 i64 i32 i64 i64 i32)
    block ;; label = @1
      block ;; label = @2
        local.get 0
        i32.reinterpret_f32
        local.tee 1
        i32.const -2139095040
        i32.add
        i32.const -2130706433
        i32.gt_u
        br_if 0 (;@2;)
        block ;; label = @3
          local.get 0
          f32.const 0x0p+0 (;=0;)
          f32.ne
          br_if 0 (;@3;)
          local.get 0
          return
        end
        block ;; label = @3
          local.get 1
          i32.const 2139095040
          i32.ne
          br_if 0 (;@3;)
          local.get 0
          return
        end
        f32.const nan (;=NaN;)
        local.set 2
        local.get 1
        i32.const 2139095040
        i32.gt_u
        br_if 1 (;@1;)
        local.get 0
        f32.const 0x1p+23 (;=8388608;)
        f32.mul
        i32.reinterpret_f32
        i32.const -192937984
        i32.add
        local.set 1
      end
      i64.const 3221225472
      i64.const 3221225472
      local.get 1
      i32.const 16
      i32.shr_u
      i32.const 254
      i32.and
      i64.load16_u offset=1048848
      i64.const 16
      i64.shl
      local.tee 3
      local.get 1
      i32.const 7
      i32.shl
      i32.const 2147483520
      i32.and
      local.get 1
      i32.const 8
      i32.shl
      i32.const -2147483648
      i32.or
      local.get 1
      i32.const 8388608
      i32.and
      select
      local.tee 4
      i64.extend_i32_u
      i64.mul
      i64.const 32
      i64.shr_u
      local.tee 5
      local.get 3
      i64.mul
      i64.const 32
      i64.shr_u
      i64.sub
      i64.const 4294967295
      i64.and
      local.tee 6
      local.get 3
      i64.mul
      i64.const 31
      i64.shr_u
      i64.const 4294967294
      i64.and
      local.get 6
      local.get 5
      i64.mul
      i64.const 31
      i64.shr_u
      i64.const 4294967294
      i64.and
      local.tee 3
      i64.mul
      i64.const 32
      i64.shr_u
      i64.sub
      i64.const 4294967295
      i64.and
      local.get 3
      i64.mul
      i64.const 38
      i64.shr_u
      i32.wrap_i64
      local.tee 7
      local.get 7
      i32.mul
      local.get 4
      i32.const 16
      i32.shl
      i32.sub
      local.get 7
      i32.add
      local.tee 4
      i32.const 31
      i32.shr_u
      local.get 7
      i32.add
      i32.const 8388607
      i32.and
      local.get 1
      i32.const 1
      i32.shr_u
      i32.const 532676608
      i32.add
      i32.const 2139095040
      i32.and
      i32.or
      local.tee 1
      f32.reinterpret_i32
      i32.const 8388608
      i32.const 0
      local.get 4
      local.get 1
      i32.add
      i32.const 1
      i32.add
      local.tee 1
      select
      local.get 1
      local.get 4
      i32.xor
      i32.const -2147483648
      i32.and
      i32.or
      f32.reinterpret_i32
      f32.add
      local.set 2
    end
    local.get 2
  )
  (func (;31;) (type 6) (param i32 i32 i32 i32)
    (local i32 i32 i32 i32 i32 f32)
    block ;; label = @1
      local.get 2
      i32.const 4
      i32.add
      local.get 3
      i32.gt_u
      br_if 0 (;@1;)
      local.get 1
      local.get 2
      i32.const 2
      i32.shl
      local.tee 4
      i32.add
      local.set 5
      local.get 0
      local.get 4
      i32.add
      local.set 4
      loop ;; label = @2
        local.get 5
        local.get 4
        v128.load align=1
        v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
        f32x4.max
        v128.store align=1
        local.get 5
        i32.const 16
        i32.add
        local.set 5
        local.get 4
        i32.const 16
        i32.add
        local.set 4
        local.get 2
        local.tee 6
        i32.const 4
        i32.add
        local.set 2
        local.get 6
        i32.const 8
        i32.add
        local.get 3
        i32.le_u
        br_if 0 (;@2;)
      end
    end
    block ;; label = @1
      local.get 3
      local.get 2
      i32.le_u
      br_if 0 (;@1;)
      block ;; label = @2
        local.get 3
        local.get 2
        i32.sub
        local.tee 7
        i32.const 4
        i32.lt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 1
        i32.sub
        i32.const -16
        i32.gt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 2
        i32.const 2
        i32.shl
        local.tee 4
        i32.add
        local.set 5
        local.get 1
        local.get 4
        i32.add
        local.set 4
        local.get 2
        local.get 7
        i32.const -4
        i32.and
        local.tee 8
        i32.add
        local.set 2
        local.get 8
        local.set 6
        loop ;; label = @3
          local.get 4
          v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
          local.get 5
          v128.load align=4
          f32x4.pmax
          v128.store align=4
          local.get 5
          i32.const 16
          i32.add
          local.set 5
          local.get 4
          i32.const 16
          i32.add
          local.set 4
          local.get 6
          i32.const -4
          i32.add
          local.tee 6
          br_if 0 (;@3;)
        end
        local.get 7
        local.get 8
        i32.eq
        br_if 1 (;@1;)
      end
      local.get 2
      local.set 7
      block ;; label = @2
        local.get 3
        local.get 2
        i32.sub
        i32.const 3
        i32.and
        local.tee 6
        i32.eqz
        br_if 0 (;@2;)
        local.get 2
        local.get 6
        i32.add
        local.set 7
        local.get 0
        local.get 2
        i32.const 2
        i32.shl
        local.tee 4
        i32.add
        local.set 5
        local.get 1
        local.get 4
        i32.add
        local.set 4
        loop ;; label = @3
          local.get 4
          local.get 5
          f32.load
          local.tee 9
          f32.const 0x0p+0 (;=0;)
          local.get 9
          f32.const 0x0p+0 (;=0;)
          f32.gt
          select
          f32.store
          local.get 5
          i32.const 4
          i32.add
          local.set 5
          local.get 4
          i32.const 4
          i32.add
          local.set 4
          local.get 6
          i32.const -1
          i32.add
          local.tee 6
          br_if 0 (;@3;)
        end
      end
      local.get 2
      local.get 3
      i32.sub
      i32.const -4
      i32.gt_u
      br_if 0 (;@1;)
      local.get 7
      i32.const 2
      i32.shl
      local.set 2
      local.get 3
      local.get 7
      i32.sub
      local.set 6
      loop ;; label = @2
        local.get 1
        local.get 2
        i32.add
        local.tee 5
        local.get 0
        local.get 2
        i32.add
        local.tee 4
        f32.load
        local.tee 9
        f32.const 0x0p+0 (;=0;)
        local.get 9
        f32.const 0x0p+0 (;=0;)
        f32.gt
        select
        f32.store
        local.get 5
        i32.const 4
        i32.add
        local.get 4
        i32.const 4
        i32.add
        f32.load
        local.tee 9
        f32.const 0x0p+0 (;=0;)
        local.get 9
        f32.const 0x0p+0 (;=0;)
        f32.gt
        select
        f32.store
        local.get 5
        i32.const 8
        i32.add
        local.get 4
        i32.const 8
        i32.add
        f32.load
        local.tee 9
        f32.const 0x0p+0 (;=0;)
        local.get 9
        f32.const 0x0p+0 (;=0;)
        f32.gt
        select
        f32.store
        local.get 5
        i32.const 12
        i32.add
        local.get 4
        i32.const 12
        i32.add
        f32.load
        local.tee 9
        f32.const 0x0p+0 (;=0;)
        local.get 9
        f32.const 0x0p+0 (;=0;)
        f32.gt
        select
        f32.store
        local.get 0
        i32.const 16
        i32.add
        local.set 0
        local.get 1
        i32.const 16
        i32.add
        local.set 1
        local.get 6
        i32.const -4
        i32.add
        local.tee 6
        br_if 0 (;@2;)
      end
    end
  )
  (func (;32;) (type 1) (param i32 i32 i32 i32 i32)
    (local i32 f32 f32)
    block ;; label = @1
      local.get 4
      local.get 3
      i32.le_u
      br_if 0 (;@1;)
      local.get 3
      i32.const 1
      i32.add
      local.set 5
      block ;; label = @2
        local.get 4
        local.get 3
        i32.sub
        i32.const 1
        i32.and
        i32.eqz
        br_if 0 (;@2;)
        f32.const 0x0p+0 (;=0;)
        local.set 6
        block ;; label = @3
          local.get 0
          local.get 3
          i32.const 2
          i32.shl
          local.tee 3
          i32.add
          f32.load
          f32.const 0x0p+0 (;=0;)
          f32.gt
          i32.eqz
          br_if 0 (;@3;)
          local.get 1
          local.get 3
          i32.add
          f32.load
          local.set 6
        end
        local.get 2
        local.get 3
        i32.add
        local.get 6
        f32.store
        local.get 5
        local.set 3
      end
      local.get 4
      local.get 5
      i32.eq
      br_if 0 (;@1;)
      local.get 4
      local.get 3
      i32.sub
      local.set 5
      local.get 0
      local.get 3
      i32.const 2
      i32.shl
      local.tee 4
      i32.add
      local.set 3
      local.get 1
      local.get 4
      i32.add
      local.set 0
      local.get 2
      local.get 4
      i32.add
      local.set 4
      loop ;; label = @2
        f32.const 0x0p+0 (;=0;)
        local.set 6
        f32.const 0x0p+0 (;=0;)
        local.set 7
        block ;; label = @3
          local.get 3
          f32.load
          f32.const 0x0p+0 (;=0;)
          f32.gt
          i32.eqz
          br_if 0 (;@3;)
          local.get 0
          f32.load
          local.set 7
        end
        local.get 4
        local.get 7
        f32.store
        block ;; label = @3
          local.get 3
          i32.const 4
          i32.add
          f32.load
          f32.const 0x0p+0 (;=0;)
          f32.gt
          i32.eqz
          br_if 0 (;@3;)
          local.get 0
          i32.const 4
          i32.add
          f32.load
          local.set 6
        end
        local.get 4
        i32.const 4
        i32.add
        local.get 6
        f32.store
        local.get 3
        i32.const 8
        i32.add
        local.set 3
        local.get 0
        i32.const 8
        i32.add
        local.set 0
        local.get 4
        i32.const 8
        i32.add
        local.set 4
        local.get 5
        i32.const -2
        i32.add
        local.tee 5
        br_if 0 (;@2;)
      end
    end
  )
  (func (;33;) (type 14) (param i32 i32 i32 i32 i32 f32 i32 i32)
    (local f32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 f32 i32 v128 f32 i32 i32 i32)
    block ;; label = @1
      local.get 6
      local.get 7
      i32.ge_u
      br_if 0 (;@1;)
      local.get 4
      i32.eqz
      br_if 0 (;@1;)
      f32.const 0x1p+0 (;=1;)
      local.get 4
      f32.convert_i32_u
      f32.div
      local.set 8
      local.get 1
      i32.const -1
      i32.xor
      local.get 6
      local.get 4
      i32.mul
      i32.const 2
      i32.shl
      local.tee 9
      local.get 2
      i32.add
      local.tee 10
      i32.add
      local.set 11
      local.get 4
      i32.const 2
      i32.shl
      local.set 12
      local.get 4
      i32.const 1
      i32.and
      local.set 13
      local.get 4
      i32.const -4
      i32.and
      local.set 14
      local.get 4
      i32.const 3
      i32.and
      local.set 15
      local.get 0
      local.get 9
      i32.add
      local.set 16
      i32.const 0
      local.set 17
      local.get 4
      i32.const 4
      i32.lt_u
      local.set 18
      local.get 0
      local.get 2
      i32.sub
      i32.const -16
      i32.gt_u
      local.set 19
      loop ;; label = @2
        f32.const 0x0p+0 (;=0;)
        local.set 20
        i32.const 0
        local.set 21
        block ;; label = @3
          block ;; label = @4
            local.get 18
            br_if 0 (;@4;)
            f32.const 0x0p+0 (;=0;)
            local.set 20
            i32.const 0
            local.set 21
            local.get 16
            local.set 9
            loop ;; label = @5
              local.get 20
              local.get 9
              v128.load align=4
              local.tee 22
              local.get 22
              f32x4.mul
              local.tee 22
              f32x4.extract_lane 0
              f32.add
              local.get 22
              f32x4.extract_lane 1
              f32.add
              local.get 22
              f32x4.extract_lane 2
              f32.add
              local.get 22
              f32x4.extract_lane 3
              f32.add
              local.set 20
              local.get 9
              i32.const 16
              i32.add
              local.set 9
              local.get 14
              local.get 21
              i32.const 4
              i32.add
              local.tee 21
              i32.ne
              br_if 0 (;@5;)
            end
            local.get 15
            i32.eqz
            br_if 1 (;@3;)
          end
          local.get 16
          local.get 21
          i32.const 2
          i32.shl
          i32.add
          local.set 9
          local.get 15
          local.set 21
          loop ;; label = @4
            local.get 20
            local.get 9
            f32.load
            local.tee 23
            local.get 23
            f32.mul
            f32.add
            local.set 20
            local.get 9
            i32.const 4
            i32.add
            local.set 9
            local.get 21
            i32.const -1
            i32.add
            local.tee 21
            br_if 0 (;@4;)
          end
        end
        f32.const 0x1p+0 (;=1;)
        local.get 5
        local.get 8
        local.get 20
        f32.mul
        f32.add
        call 30
        f32.div
        local.set 20
        i32.const 0
        local.set 21
        block ;; label = @3
          block ;; label = @4
            local.get 18
            br_if 0 (;@4;)
            local.get 19
            br_if 0 (;@4;)
            local.get 12
            local.get 17
            i32.mul
            local.get 11
            i32.add
            i32.const 15
            i32.lt_u
            br_if 0 (;@4;)
            local.get 20
            f32x4.splat
            local.set 22
            i32.const 0
            local.set 9
            local.get 14
            local.set 21
            loop ;; label = @5
              local.get 10
              local.get 9
              i32.add
              local.get 22
              local.get 16
              local.get 9
              i32.add
              v128.load align=4
              f32x4.mul
              local.get 1
              local.get 9
              i32.add
              v128.load align=4
              f32x4.mul
              v128.store align=4
              local.get 9
              i32.const 16
              i32.add
              local.set 9
              local.get 21
              i32.const -4
              i32.add
              local.tee 21
              br_if 0 (;@5;)
            end
            local.get 14
            local.set 21
            local.get 4
            local.get 14
            i32.eq
            br_if 1 (;@3;)
          end
          local.get 21
          i32.const 1
          i32.or
          local.set 9
          block ;; label = @4
            local.get 13
            i32.eqz
            br_if 0 (;@4;)
            local.get 2
            local.get 21
            local.get 6
            local.get 4
            i32.mul
            i32.add
            i32.const 2
            i32.shl
            local.tee 24
            i32.add
            local.get 20
            local.get 0
            local.get 24
            i32.add
            f32.load
            f32.mul
            local.get 1
            local.get 21
            i32.const 2
            i32.shl
            i32.add
            f32.load
            f32.mul
            f32.store
            local.get 9
            local.set 21
          end
          local.get 4
          local.get 9
          i32.eq
          br_if 0 (;@3;)
          local.get 21
          i32.const 2
          i32.shl
          local.set 9
          local.get 4
          local.get 21
          i32.sub
          local.set 21
          loop ;; label = @4
            local.get 10
            local.get 9
            i32.add
            local.tee 24
            local.get 20
            local.get 16
            local.get 9
            i32.add
            local.tee 25
            f32.load
            f32.mul
            local.get 1
            local.get 9
            i32.add
            local.tee 26
            f32.load
            f32.mul
            f32.store
            local.get 24
            i32.const 4
            i32.add
            local.get 20
            local.get 25
            i32.const 4
            i32.add
            f32.load
            f32.mul
            local.get 26
            i32.const 4
            i32.add
            f32.load
            f32.mul
            f32.store
            local.get 9
            i32.const 8
            i32.add
            local.set 9
            local.get 21
            i32.const -2
            i32.add
            local.tee 21
            br_if 0 (;@4;)
          end
        end
        local.get 10
        local.get 12
        i32.add
        local.set 10
        local.get 16
        local.get 12
        i32.add
        local.set 16
        local.get 17
        i32.const 1
        i32.add
        local.set 17
        local.get 6
        i32.const 1
        i32.add
        local.tee 6
        local.get 7
        i32.ne
        br_if 0 (;@2;)
      end
    end
  )
  (func (;34;) (type 1) (param i32 i32 i32 i32 i32)
    (local i32 i32 i32 i32 i32 i32 v128 f32)
    block ;; label = @1
      local.get 4
      local.get 3
      i32.le_u
      br_if 0 (;@1;)
      block ;; label = @2
        local.get 4
        local.get 3
        i32.sub
        local.tee 5
        i32.const 8
        i32.lt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 2
        i32.sub
        i32.const -16
        i32.gt_u
        br_if 0 (;@2;)
        local.get 1
        local.get 2
        i32.sub
        i32.const -16
        i32.gt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 3
        i32.const 2
        i32.shl
        local.tee 6
        i32.add
        local.set 7
        local.get 1
        local.get 6
        i32.add
        local.set 8
        local.get 2
        local.get 6
        i32.add
        local.set 6
        local.get 3
        local.get 5
        i32.const -4
        i32.and
        local.tee 9
        i32.add
        local.set 3
        local.get 9
        local.set 10
        loop ;; label = @3
          local.get 6
          local.get 7
          v128.load align=4
          local.tee 11
          local.get 11
          local.get 11
          local.get 8
          v128.load align=4
          v128.const i32x4 0xbf000000 0xbf000000 0xbf000000 0xbf000000
          f32x4.mul
          f32x4.mul
          f32x4.mul
          f32x4.mul
          v128.store align=4
          local.get 7
          i32.const 16
          i32.add
          local.set 7
          local.get 8
          i32.const 16
          i32.add
          local.set 8
          local.get 6
          i32.const 16
          i32.add
          local.set 6
          local.get 10
          i32.const -4
          i32.add
          local.tee 10
          br_if 0 (;@3;)
        end
        local.get 5
        local.get 9
        i32.eq
        br_if 1 (;@1;)
      end
      local.get 3
      i32.const 1
      i32.add
      local.set 7
      block ;; label = @2
        local.get 4
        local.get 3
        i32.sub
        i32.const 1
        i32.and
        i32.eqz
        br_if 0 (;@2;)
        local.get 2
        local.get 3
        i32.const 2
        i32.shl
        local.tee 8
        i32.add
        local.get 0
        local.get 8
        i32.add
        f32.load
        local.tee 12
        local.get 12
        local.get 12
        local.get 1
        local.get 8
        i32.add
        f32.load
        f32.const -0x1p-1 (;=-0.5;)
        f32.mul
        f32.mul
        f32.mul
        f32.mul
        f32.store
        local.get 7
        local.set 3
      end
      local.get 4
      local.get 7
      i32.eq
      br_if 0 (;@1;)
      local.get 4
      local.get 3
      i32.sub
      local.set 10
      local.get 0
      local.get 3
      i32.const 2
      i32.shl
      local.tee 6
      i32.add
      local.set 7
      local.get 1
      local.get 6
      i32.add
      local.set 8
      local.get 2
      local.get 6
      i32.add
      local.set 6
      loop ;; label = @2
        local.get 6
        local.get 7
        f32.load
        local.tee 12
        local.get 12
        local.get 12
        local.get 8
        f32.load
        f32.const -0x1p-1 (;=-0.5;)
        f32.mul
        f32.mul
        f32.mul
        f32.mul
        f32.store
        local.get 6
        i32.const 4
        i32.add
        local.get 7
        i32.const 4
        i32.add
        f32.load
        local.tee 12
        local.get 12
        local.get 12
        local.get 8
        i32.const 4
        i32.add
        f32.load
        f32.const -0x1p-1 (;=-0.5;)
        f32.mul
        f32.mul
        f32.mul
        f32.mul
        f32.store
        local.get 7
        i32.const 8
        i32.add
        local.set 7
        local.get 8
        i32.const 8
        i32.add
        local.set 8
        local.get 6
        i32.const 8
        i32.add
        local.set 6
        local.get 10
        i32.const -2
        i32.add
        local.tee 10
        br_if 0 (;@2;)
      end
    end
  )
  (func (;35;) (type 6) (param i32 i32 i32 i32)
    (local i32 i32 i32 v128)
    block ;; label = @1
      local.get 2
      i32.const 4
      i32.add
      local.get 3
      i32.gt_u
      br_if 0 (;@1;)
      local.get 1
      local.get 2
      i32.const 2
      i32.shl
      local.tee 4
      i32.add
      local.set 5
      local.get 0
      local.get 4
      i32.add
      local.set 4
      loop ;; label = @2
        local.get 5
        v128.const i32x4 0x3f800000 0x3f800000 0x3f800000 0x3f800000
        local.get 4
        v128.load align=1
        f32x4.sqrt
        f32x4.div
        v128.store align=1
        local.get 5
        i32.const 16
        i32.add
        local.set 5
        local.get 4
        i32.const 16
        i32.add
        local.set 4
        local.get 2
        local.tee 6
        i32.const 4
        i32.add
        local.set 2
        local.get 6
        i32.const 8
        i32.add
        local.get 3
        i32.le_u
        br_if 0 (;@2;)
      end
    end
    block ;; label = @1
      local.get 3
      local.get 2
      i32.le_u
      br_if 0 (;@1;)
      local.get 2
      i32.const 1
      i32.add
      local.set 5
      block ;; label = @2
        local.get 3
        local.get 2
        i32.sub
        i32.const 1
        i32.and
        i32.eqz
        br_if 0 (;@2;)
        local.get 1
        local.get 2
        i32.const 2
        i32.shl
        local.tee 4
        i32.add
        f32.const 0x1p+0 (;=1;)
        local.get 0
        local.get 4
        i32.add
        v128.const i32x4 0x00000000 0x3f800000 0x3f800000 0x3f800000
        v128.load32_lane 0
        f32x4.sqrt
        f32x4.extract_lane 0
        f32.div
        f32.store
        local.get 5
        local.set 2
      end
      local.get 3
      local.get 5
      i32.eq
      br_if 0 (;@1;)
      local.get 3
      local.get 2
      i32.sub
      local.set 6
      local.get 0
      local.get 2
      i32.const 2
      i32.shl
      local.tee 4
      i32.add
      local.set 5
      local.get 1
      local.get 4
      i32.add
      local.set 4
      loop ;; label = @2
        local.get 4
        f32.const 0x1p+0 (;=1;)
        local.get 5
        v128.const i32x4 0x3f800000 0x3f800000 0x3f800000 0x3f800000
        local.tee 7
        v128.load32_lane 0
        f32x4.sqrt
        f32x4.extract_lane 0
        f32.div
        f32.store
        local.get 4
        i32.const 4
        i32.add
        f32.const 0x1p+0 (;=1;)
        local.get 5
        i32.const 4
        i32.add
        local.get 7
        v128.load32_lane 0
        f32x4.sqrt
        f32x4.extract_lane 0
        f32.div
        f32.store
        local.get 5
        i32.const 8
        i32.add
        local.set 5
        local.get 4
        i32.const 8
        i32.add
        local.set 4
        local.get 6
        i32.const -2
        i32.add
        local.tee 6
        br_if 0 (;@2;)
      end
    end
  )
  (func (;36;) (type 6) (param i32 i32 i32 i32)
    (local i32)
    block ;; label = @1
      local.get 3
      local.get 2
      i32.le_u
      br_if 0 (;@1;)
      local.get 3
      local.get 2
      i32.sub
      local.set 4
      local.get 0
      local.get 2
      i32.const 2
      i32.shl
      local.tee 3
      i32.add
      local.set 2
      local.get 1
      local.get 3
      i32.add
      local.set 3
      loop ;; label = @2
        local.get 3
        f32.const 0x1p+0 (;=1;)
        local.get 2
        f32.load
        f32.neg
        call 17
        f32.const 0x1p+0 (;=1;)
        f32.add
        f32.div
        f32.store
        local.get 2
        i32.const 4
        i32.add
        local.set 2
        local.get 3
        i32.const 4
        i32.add
        local.set 3
        local.get 4
        i32.const -1
        i32.add
        local.tee 4
        br_if 0 (;@2;)
      end
    end
  )
  (func (;37;) (type 1) (param i32 i32 i32 i32 i32)
    (local i32 i32 i32 i32 i32 i32 v128 f32)
    block ;; label = @1
      local.get 4
      local.get 3
      i32.le_u
      br_if 0 (;@1;)
      block ;; label = @2
        local.get 4
        local.get 3
        i32.sub
        local.tee 5
        i32.const 8
        i32.lt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 2
        i32.sub
        i32.const -16
        i32.gt_u
        br_if 0 (;@2;)
        local.get 1
        local.get 2
        i32.sub
        i32.const -16
        i32.gt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 3
        i32.const 2
        i32.shl
        local.tee 6
        i32.add
        local.set 7
        local.get 1
        local.get 6
        i32.add
        local.set 8
        local.get 2
        local.get 6
        i32.add
        local.set 6
        local.get 3
        local.get 5
        i32.const -4
        i32.and
        local.tee 9
        i32.add
        local.set 3
        local.get 9
        local.set 10
        loop ;; label = @3
          local.get 6
          v128.const i32x4 0x3f800000 0x3f800000 0x3f800000 0x3f800000
          local.get 7
          v128.load align=4
          local.tee 11
          f32x4.sub
          local.get 11
          local.get 8
          v128.load align=4
          f32x4.mul
          f32x4.mul
          v128.store align=4
          local.get 7
          i32.const 16
          i32.add
          local.set 7
          local.get 8
          i32.const 16
          i32.add
          local.set 8
          local.get 6
          i32.const 16
          i32.add
          local.set 6
          local.get 10
          i32.const -4
          i32.add
          local.tee 10
          br_if 0 (;@3;)
        end
        local.get 5
        local.get 9
        i32.eq
        br_if 1 (;@1;)
      end
      local.get 3
      i32.const 1
      i32.add
      local.set 7
      block ;; label = @2
        local.get 4
        local.get 3
        i32.sub
        i32.const 1
        i32.and
        i32.eqz
        br_if 0 (;@2;)
        local.get 2
        local.get 3
        i32.const 2
        i32.shl
        local.tee 8
        i32.add
        f32.const 0x1p+0 (;=1;)
        local.get 0
        local.get 8
        i32.add
        f32.load
        local.tee 12
        f32.sub
        local.get 12
        local.get 1
        local.get 8
        i32.add
        f32.load
        f32.mul
        f32.mul
        f32.store
        local.get 7
        local.set 3
      end
      local.get 4
      local.get 7
      i32.eq
      br_if 0 (;@1;)
      local.get 4
      local.get 3
      i32.sub
      local.set 10
      local.get 0
      local.get 3
      i32.const 2
      i32.shl
      local.tee 6
      i32.add
      local.set 7
      local.get 1
      local.get 6
      i32.add
      local.set 8
      local.get 2
      local.get 6
      i32.add
      local.set 6
      loop ;; label = @2
        local.get 6
        f32.const 0x1p+0 (;=1;)
        local.get 7
        f32.load
        local.tee 12
        f32.sub
        local.get 12
        local.get 8
        f32.load
        f32.mul
        f32.mul
        f32.store
        local.get 6
        i32.const 4
        i32.add
        f32.const 0x1p+0 (;=1;)
        local.get 7
        i32.const 4
        i32.add
        f32.load
        local.tee 12
        f32.sub
        local.get 12
        local.get 8
        i32.const 4
        i32.add
        f32.load
        f32.mul
        f32.mul
        f32.store
        local.get 7
        i32.const 8
        i32.add
        local.set 7
        local.get 8
        i32.const 8
        i32.add
        local.set 8
        local.get 6
        i32.const 8
        i32.add
        local.set 6
        local.get 10
        i32.const -2
        i32.add
        local.tee 10
        br_if 0 (;@2;)
      end
    end
  )
  (func (;38;) (type 6) (param i32 i32 i32 i32)
    (local i32 f32)
    block ;; label = @1
      local.get 3
      local.get 2
      i32.le_u
      br_if 0 (;@1;)
      local.get 3
      local.get 2
      i32.sub
      local.set 4
      local.get 0
      local.get 2
      i32.const 2
      i32.shl
      local.tee 3
      i32.add
      local.set 2
      local.get 1
      local.get 3
      i32.add
      local.set 3
      loop ;; label = @2
        local.get 2
        f32.load
        local.set 5
        local.get 3
        local.get 5
        f32.const 0x1p+0 (;=1;)
        local.get 5
        f32.neg
        call 17
        f32.const 0x1p+0 (;=1;)
        f32.add
        f32.div
        f32.mul
        f32.store
        local.get 2
        i32.const 4
        i32.add
        local.set 2
        local.get 3
        i32.const 4
        i32.add
        local.set 3
        local.get 4
        i32.const -1
        i32.add
        local.tee 4
        br_if 0 (;@2;)
      end
    end
  )
  (func (;39;) (type 1) (param i32 i32 i32 i32 i32)
    (local i32 i32 f32 f32)
    block ;; label = @1
      local.get 4
      local.get 3
      i32.le_u
      br_if 0 (;@1;)
      local.get 4
      local.get 3
      i32.sub
      local.set 5
      local.get 0
      local.get 3
      i32.const 2
      i32.shl
      local.tee 6
      i32.add
      local.set 3
      local.get 1
      local.get 6
      i32.add
      local.set 4
      local.get 2
      local.get 6
      i32.add
      local.set 2
      loop ;; label = @2
        local.get 3
        f32.load
        local.tee 7
        f32.neg
        call 17
        local.set 8
        local.get 2
        local.get 4
        f32.load
        f32.const 0x1p+0 (;=1;)
        local.get 8
        f32.const 0x1p+0 (;=1;)
        f32.add
        f32.div
        local.tee 8
        f32.mul
        local.get 7
        f32.const 0x1p+0 (;=1;)
        local.get 8
        f32.sub
        f32.mul
        f32.const 0x1p+0 (;=1;)
        f32.add
        f32.mul
        f32.store
        local.get 3
        i32.const 4
        i32.add
        local.set 3
        local.get 4
        i32.const 4
        i32.add
        local.set 4
        local.get 2
        i32.const 4
        i32.add
        local.set 2
        local.get 5
        i32.const -1
        i32.add
        local.tee 5
        br_if 0 (;@2;)
      end
    end
  )
  (func (;40;) (type 6) (param i32 i32 i32 i32)
    (local i32 i32 f32 f64 f64 i32 f64)
    global.get 0
    i32.const 16
    i32.sub
    local.tee 4
    global.set 0
    block ;; label = @1
      local.get 3
      local.get 2
      i32.le_u
      br_if 0 (;@1;)
      local.get 3
      local.get 2
      i32.sub
      local.set 5
      local.get 0
      local.get 2
      i32.const 2
      i32.shl
      local.tee 2
      i32.add
      local.set 3
      local.get 1
      local.get 2
      i32.add
      local.set 1
      loop ;; label = @2
        local.get 3
        f32.load
        local.tee 6
        f64.promote_f32
        local.set 7
        block ;; label = @3
          block ;; label = @4
            local.get 6
            i32.reinterpret_f32
            local.tee 0
            i32.const 2147483647
            i32.and
            local.tee 2
            i32.const 1061752795
            i32.lt_u
            br_if 0 (;@4;)
            block ;; label = @5
              local.get 2
              i32.const 1081824210
              i32.lt_u
              br_if 0 (;@5;)
              block ;; label = @6
                local.get 2
                i32.const 1088565718
                i32.lt_u
                br_if 0 (;@6;)
                block ;; label = @7
                  block ;; label = @8
                    block ;; label = @9
                      block ;; label = @10
                        block ;; label = @11
                          local.get 2
                          i32.const 2139095039
                          i32.gt_u
                          br_if 0 (;@11;)
                          local.get 4
                          i64.const 0
                          i64.store offset=8
                          block ;; label = @12
                            block ;; label = @13
                              local.get 2
                              i32.const 1305022426
                              i32.gt_u
                              br_if 0 (;@13;)
                              local.get 7
                              local.get 7
                              f64.const 0x1.45f306dc9c883p-1 (;=0.6366197723675814;)
                              f64.mul
                              f64.const 0x1.8p+52 (;=6755399441055744;)
                              f64.add
                              f64.const -0x1.8p+52 (;=-6755399441055744;)
                              f64.add
                              local.tee 8
                              f64.const -0x1.921fb5p+0 (;=-1.5707963109016418;)
                              f64.mul
                              f64.add
                              local.get 8
                              f64.const -0x1.110b4611a6263p-26 (;=-0.000000015893254773528196;)
                              f64.mul
                              f64.add
                              local.set 7
                              local.get 8
                              i32.trunc_sat_f64_s
                              local.set 2
                              br 1 (;@12;)
                            end
                            local.get 2
                            local.get 2
                            i32.const 23
                            i32.shr_u
                            i32.const -150
                            i32.add
                            local.tee 9
                            i32.const 23
                            i32.shl
                            i32.sub
                            f32.reinterpret_i32
                            f64.promote_f32
                            local.get 4
                            i32.const 8
                            i32.add
                            local.get 9
                            call 41
                            local.set 2
                            block ;; label = @13
                              local.get 0
                              i32.const 0
                              i32.lt_s
                              br_if 0 (;@13;)
                              local.get 4
                              f64.load offset=8
                              local.set 7
                              br 1 (;@12;)
                            end
                            i32.const 0
                            local.get 2
                            i32.sub
                            local.set 2
                            local.get 4
                            f64.load offset=8
                            f64.neg
                            local.set 7
                          end
                          local.get 2
                          i32.const 3
                          i32.and
                          br_table 2 (;@9;) 3 (;@8;) 4 (;@7;) 1 (;@10;) 2 (;@9;)
                        end
                        local.get 6
                        local.get 6
                        f32.sub
                        local.set 6
                        br 7 (;@3;)
                      end
                      local.get 7
                      local.get 7
                      f64.mul
                      local.tee 7
                      f64.const -0x1.ffffffd0c5e81p-2 (;=-0.499999997251031;)
                      f64.mul
                      f64.const 0x1p+0 (;=1;)
                      f64.add
                      local.get 7
                      local.get 7
                      f64.mul
                      local.tee 8
                      f64.const 0x1.55553e1053a42p-5 (;=0.04166662332373906;)
                      f64.mul
                      f64.add
                      local.get 7
                      local.get 8
                      f64.mul
                      local.get 7
                      f64.const 0x1.99342e0ee5069p-16 (;=0.00002439044879627741;)
                      f64.mul
                      f64.const -0x1.6c087e80f1e27p-10 (;=-0.001388676377460993;)
                      f64.add
                      f64.mul
                      f64.add
                      f32.demote_f64
                      f32.neg
                      local.set 6
                      br 6 (;@3;)
                    end
                    local.get 7
                    local.get 7
                    local.get 7
                    f64.mul
                    local.tee 8
                    f64.mul
                    local.tee 10
                    local.get 8
                    local.get 8
                    f64.mul
                    f64.mul
                    local.get 8
                    f64.const 0x1.6cd878c3b46a7p-19 (;=0.000002718311493989822;)
                    f64.mul
                    f64.const -0x1.a00f9e2cae774p-13 (;=-0.00019839334836096632;)
                    f64.add
                    f64.mul
                    local.get 7
                    local.get 10
                    local.get 8
                    f64.const 0x1.11110896efbb2p-7 (;=0.008333329385889463;)
                    f64.mul
                    f64.const -0x1.5555554cbac77p-3 (;=-0.16666666641626524;)
                    f64.add
                    f64.mul
                    f64.add
                    f64.add
                    f32.demote_f64
                    local.set 6
                    br 5 (;@3;)
                  end
                  local.get 7
                  local.get 7
                  f64.mul
                  local.tee 7
                  f64.const -0x1.ffffffd0c5e81p-2 (;=-0.499999997251031;)
                  f64.mul
                  f64.const 0x1p+0 (;=1;)
                  f64.add
                  local.get 7
                  local.get 7
                  f64.mul
                  local.tee 8
                  f64.const 0x1.55553e1053a42p-5 (;=0.04166662332373906;)
                  f64.mul
                  f64.add
                  local.get 7
                  local.get 8
                  f64.mul
                  local.get 7
                  f64.const 0x1.99342e0ee5069p-16 (;=0.00002439044879627741;)
                  f64.mul
                  f64.const -0x1.6c087e80f1e27p-10 (;=-0.001388676377460993;)
                  f64.add
                  f64.mul
                  f64.add
                  f32.demote_f64
                  local.set 6
                  br 4 (;@3;)
                end
                local.get 7
                local.get 7
                f64.mul
                local.tee 8
                local.get 7
                f64.neg
                f64.mul
                local.tee 10
                local.get 8
                local.get 8
                f64.mul
                f64.mul
                local.get 8
                f64.const 0x1.6cd878c3b46a7p-19 (;=0.000002718311493989822;)
                f64.mul
                f64.const -0x1.a00f9e2cae774p-13 (;=-0.00019839334836096632;)
                f64.add
                f64.mul
                local.get 10
                local.get 8
                f64.const 0x1.11110896efbb2p-7 (;=0.008333329385889463;)
                f64.mul
                f64.const -0x1.5555554cbac77p-3 (;=-0.16666666641626524;)
                f64.add
                f64.mul
                local.get 7
                f64.sub
                f64.add
                f32.demote_f64
                local.set 6
                br 3 (;@3;)
              end
              block ;; label = @6
                local.get 2
                i32.const 1085271520
                i32.lt_u
                br_if 0 (;@6;)
                f64.const -0x1.921fb54442d18p+2 (;=-6.283185307179586;)
                f64.const 0x1.921fb54442d18p+2 (;=6.283185307179586;)
                local.get 0
                i32.const -1
                i32.gt_s
                select
                local.get 7
                f64.add
                local.tee 8
                local.get 8
                local.get 8
                f64.mul
                local.tee 7
                f64.mul
                local.tee 10
                local.get 7
                local.get 7
                f64.mul
                f64.mul
                local.get 7
                f64.const 0x1.6cd878c3b46a7p-19 (;=0.000002718311493989822;)
                f64.mul
                f64.const -0x1.a00f9e2cae774p-13 (;=-0.00019839334836096632;)
                f64.add
                f64.mul
                local.get 8
                local.get 10
                local.get 7
                f64.const 0x1.11110896efbb2p-7 (;=0.008333329385889463;)
                f64.mul
                f64.const -0x1.5555554cbac77p-3 (;=-0.16666666641626524;)
                f64.add
                f64.mul
                f64.add
                f64.add
                f32.demote_f64
                local.set 6
                br 3 (;@3;)
              end
              block ;; label = @6
                local.get 0
                i32.const 0
                i32.lt_s
                br_if 0 (;@6;)
                local.get 7
                f64.const -0x1.2d97c7f3321d2p+2 (;=-4.71238898038469;)
                f64.add
                local.tee 7
                local.get 7
                f64.mul
                local.tee 7
                f64.const -0x1.ffffffd0c5e81p-2 (;=-0.499999997251031;)
                f64.mul
                f64.const 0x1p+0 (;=1;)
                f64.add
                local.get 7
                local.get 7
                f64.mul
                local.tee 8
                f64.const 0x1.55553e1053a42p-5 (;=0.04166662332373906;)
                f64.mul
                f64.add
                local.get 7
                local.get 8
                f64.mul
                local.get 7
                f64.const 0x1.99342e0ee5069p-16 (;=0.00002439044879627741;)
                f64.mul
                f64.const -0x1.6c087e80f1e27p-10 (;=-0.001388676377460993;)
                f64.add
                f64.mul
                f64.add
                f32.demote_f64
                f32.neg
                local.set 6
                br 3 (;@3;)
              end
              local.get 7
              f64.const 0x1.2d97c7f3321d2p+2 (;=4.71238898038469;)
              f64.add
              local.tee 7
              local.get 7
              f64.mul
              local.tee 7
              f64.const -0x1.ffffffd0c5e81p-2 (;=-0.499999997251031;)
              f64.mul
              f64.const 0x1p+0 (;=1;)
              f64.add
              local.get 7
              local.get 7
              f64.mul
              local.tee 8
              f64.const 0x1.55553e1053a42p-5 (;=0.04166662332373906;)
              f64.mul
              f64.add
              local.get 7
              local.get 8
              f64.mul
              local.get 7
              f64.const 0x1.99342e0ee5069p-16 (;=0.00002439044879627741;)
              f64.mul
              f64.const -0x1.6c087e80f1e27p-10 (;=-0.001388676377460993;)
              f64.add
              f64.mul
              f64.add
              f32.demote_f64
              local.set 6
              br 2 (;@3;)
            end
            block ;; label = @5
              local.get 2
              i32.const 1075235812
              i32.lt_u
              br_if 0 (;@5;)
              f64.const -0x1.921fb54442d18p+1 (;=-3.141592653589793;)
              f64.const 0x1.921fb54442d18p+1 (;=3.141592653589793;)
              local.get 0
              i32.const -1
              i32.gt_s
              select
              local.get 7
              f64.add
              local.tee 8
              local.get 8
              f64.mul
              local.tee 7
              local.get 8
              f64.neg
              f64.mul
              local.tee 10
              local.get 7
              local.get 7
              f64.mul
              f64.mul
              local.get 7
              f64.const 0x1.6cd878c3b46a7p-19 (;=0.000002718311493989822;)
              f64.mul
              f64.const -0x1.a00f9e2cae774p-13 (;=-0.00019839334836096632;)
              f64.add
              f64.mul
              local.get 10
              local.get 7
              f64.const 0x1.11110896efbb2p-7 (;=0.008333329385889463;)
              f64.mul
              f64.const -0x1.5555554cbac77p-3 (;=-0.16666666641626524;)
              f64.add
              f64.mul
              local.get 8
              f64.sub
              f64.add
              f32.demote_f64
              local.set 6
              br 2 (;@3;)
            end
            block ;; label = @5
              local.get 0
              i32.const 0
              i32.lt_s
              br_if 0 (;@5;)
              local.get 7
              f64.const -0x1.921fb54442d18p+0 (;=-1.5707963267948966;)
              f64.add
              local.tee 7
              local.get 7
              f64.mul
              local.tee 7
              f64.const -0x1.ffffffd0c5e81p-2 (;=-0.499999997251031;)
              f64.mul
              f64.const 0x1p+0 (;=1;)
              f64.add
              local.get 7
              local.get 7
              f64.mul
              local.tee 8
              f64.const 0x1.55553e1053a42p-5 (;=0.04166662332373906;)
              f64.mul
              f64.add
              local.get 7
              local.get 8
              f64.mul
              local.get 7
              f64.const 0x1.99342e0ee5069p-16 (;=0.00002439044879627741;)
              f64.mul
              f64.const -0x1.6c087e80f1e27p-10 (;=-0.001388676377460993;)
              f64.add
              f64.mul
              f64.add
              f32.demote_f64
              local.set 6
              br 2 (;@3;)
            end
            local.get 7
            f64.const 0x1.921fb54442d18p+0 (;=1.5707963267948966;)
            f64.add
            local.tee 7
            local.get 7
            f64.mul
            local.tee 7
            f64.const -0x1.ffffffd0c5e81p-2 (;=-0.499999997251031;)
            f64.mul
            f64.const 0x1p+0 (;=1;)
            f64.add
            local.get 7
            local.get 7
            f64.mul
            local.tee 8
            f64.const 0x1.55553e1053a42p-5 (;=0.04166662332373906;)
            f64.mul
            f64.add
            local.get 7
            local.get 8
            f64.mul
            local.get 7
            f64.const 0x1.99342e0ee5069p-16 (;=0.00002439044879627741;)
            f64.mul
            f64.const -0x1.6c087e80f1e27p-10 (;=-0.001388676377460993;)
            f64.add
            f64.mul
            f64.add
            f32.demote_f64
            f32.neg
            local.set 6
            br 1 (;@3;)
          end
          block ;; label = @4
            local.get 2
            i32.const 964689920
            i32.lt_u
            br_if 0 (;@4;)
            local.get 7
            local.get 7
            f64.mul
            local.tee 8
            local.get 7
            f64.mul
            local.tee 10
            local.get 8
            local.get 8
            f64.mul
            f64.mul
            local.get 8
            f64.const 0x1.6cd878c3b46a7p-19 (;=0.000002718311493989822;)
            f64.mul
            f64.const -0x1.a00f9e2cae774p-13 (;=-0.00019839334836096632;)
            f64.add
            f64.mul
            local.get 10
            local.get 8
            f64.const 0x1.11110896efbb2p-7 (;=0.008333329385889463;)
            f64.mul
            f64.const -0x1.5555554cbac77p-3 (;=-0.16666666641626524;)
            f64.add
            f64.mul
            local.get 7
            f64.add
            f64.add
            f32.demote_f64
            local.set 6
            br 1 (;@3;)
          end
          local.get 4
          local.get 6
          f32.const 0x1p-120 (;=0.0000000000000000000000000000000000007523164;)
          f32.mul
          local.get 6
          f32.const 0x1p+120 (;=1329228000000000000000000000000000000;)
          f32.add
          local.get 2
          i32.const 8388608
          i32.lt_u
          select
          f32.store offset=4
          local.get 4
          f32.load offset=4
          drop
        end
        local.get 1
        local.get 6
        f32.store
        local.get 3
        i32.const 4
        i32.add
        local.set 3
        local.get 1
        i32.const 4
        i32.add
        local.set 1
        local.get 5
        i32.const -1
        i32.add
        local.tee 5
        br_if 0 (;@2;)
      end
    end
    local.get 4
    i32.const 16
    i32.add
    global.set 0
  )
  (func (;41;) (type 15) (param f64 i32 i32) (result i32)
    (local i32 i32 i32 i32 v128 v128 i32 i32 i32 f64 i32 i32 i32 i32 i32 i32 i32 i32 f64 f64 f64 i64 i64 i64 i32 i32 i32 i32 i32 i32 i32 i32 v128)
    global.get 0
    i32.const 560
    i32.sub
    local.tee 3
    global.set 0
    local.get 3
    i32.const 0
    i32.const 160
    memory.fill
    local.get 3
    i32.const 160
    i32.add
    i32.const 0
    i32.const 160
    memory.fill
    local.get 3
    i32.const 320
    i32.add
    i32.const 0
    i32.const 160
    memory.fill
    local.get 3
    i32.const 480
    i32.add
    i32.const 0
    i32.const 80
    memory.fill
    local.get 3
    local.get 2
    i32.const -3
    i32.add
    i32.const 65535
    i32.and
    i32.const 24
    i32.div_u
    local.tee 4
    i32.const 2
    i32.shl
    local.tee 5
    i32.const 1048580
    i32.add
    local.tee 6
    i32.load
    f64.convert_i32_s
    f64.store offset=8
    local.get 3
    local.get 5
    i32.load offset=1048576
    f64.convert_i32_s
    f64.store
    local.get 3
    local.get 5
    i32.load offset=1048588
    f64.convert_i32_s
    f64.store offset=24
    local.get 3
    local.get 5
    i32.load offset=1048584
    f64.convert_i32_s
    f64.store offset=16
    local.get 3
    local.get 0
    f64x2.splat
    local.tee 7
    local.get 3
    v128.load
    f64x2.mul
    v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
    local.tee 8
    f64x2.add
    v128.store offset=320
    local.get 3
    local.get 7
    local.get 3
    v128.load offset=16
    f64x2.mul
    local.get 8
    f64x2.add
    v128.store offset=336
    i32.const 47
    local.get 2
    local.get 4
    i32.const -24
    i32.mul
    i32.add
    local.tee 9
    i32.sub
    i32.const 31
    i32.and
    local.set 10
    i32.const 48
    local.get 9
    i32.sub
    i32.const 31
    i32.and
    local.set 11
    local.get 9
    i32.const 999
    i32.add
    i64.extend_i32_u
    i64.const 52
    i64.shl
    f64.reinterpret_i64
    local.set 12
    local.get 3
    i32.const 8
    i32.or
    local.set 13
    local.get 3
    i32.const 480
    i32.add
    i32.const 8
    i32.add
    local.set 14
    local.get 3
    i32.const 480
    i32.add
    i32.const -16
    i32.add
    local.set 15
    local.get 3
    i32.const 320
    i32.add
    i32.const -8
    i32.add
    local.set 16
    local.get 9
    i32.const -24
    i32.add
    local.tee 17
    i32.const -1
    i32.add
    local.set 18
    i32.const 3
    local.set 19
    loop (result i32) ;; label = @1
      local.get 3
      i32.const 320
      i32.add
      local.get 19
      local.tee 20
      i32.const 3
      i32.shl
      local.tee 5
      i32.add
      f64.load
      local.set 21
      block ;; label = @2
        local.get 20
        i32.eqz
        br_if 0 (;@2;)
        local.get 3
        i32.const 480
        i32.add
        local.set 2
        loop ;; label = @3
          local.get 2
          local.get 21
          local.get 21
          f64.const 0x1p-24 (;=0.00000005960464477539063;)
          f64.mul
          i32.trunc_sat_f64_s
          f64.convert_i32_s
          local.tee 22
          f64.const -0x1p+24 (;=-16777216;)
          f64.mul
          f64.add
          i32.trunc_sat_f64_s
          i32.store
          local.get 16
          local.get 5
          i32.add
          f64.load
          local.get 22
          f64.add
          local.set 21
          local.get 2
          i32.const 4
          i32.add
          local.set 2
          local.get 5
          i32.const -8
          i32.add
          local.tee 5
          br_if 0 (;@3;)
        end
      end
      block ;; label = @2
        block ;; label = @3
          local.get 21
          local.get 12
          f64.mul
          local.tee 22
          f64.const 0x1p-3 (;=0.125;)
          f64.mul
          local.tee 23
          i64.reinterpret_f64
          local.tee 24
          i64.const 52
          i64.shr_u
          i32.wrap_i64
          i32.const 2047
          i32.and
          local.tee 5
          i32.const 1074
          i32.le_u
          br_if 0 (;@3;)
          local.get 23
          local.set 21
          br 1 (;@2;)
        end
        block ;; label = @3
          local.get 5
          i32.const 1022
          i32.gt_u
          br_if 0 (;@3;)
          f64.const 0x0p+0 (;=0;)
          local.set 21
          local.get 24
          i64.const -1
          i64.gt_s
          br_if 1 (;@2;)
          local.get 23
          f64.const -0x1p+0 (;=-1;)
          local.get 23
          f64.const 0x0p+0 (;=0;)
          f64.eq
          select
          local.set 21
          br 1 (;@2;)
        end
        local.get 23
        local.set 21
        i64.const 4503599627370495
        local.get 5
        i32.const -1023
        i32.add
        i64.extend_i32_u
        local.tee 25
        i64.shr_u
        local.tee 26
        local.get 24
        i64.and
        i64.eqz
        br_if 0 (;@2;)
        local.get 24
        i64.const 63
        i64.shr_s
        local.get 26
        i64.and
        local.get 24
        i64.add
        i64.const -4503599627370496
        local.get 25
        i64.shr_s
        i64.and
        f64.reinterpret_i64
        local.set 21
      end
      local.get 22
      local.get 21
      f64.const -0x1p+3 (;=-8;)
      f64.mul
      f64.add
      local.tee 21
      local.get 21
      i32.trunc_sat_f64_s
      local.tee 27
      f64.convert_i32_s
      f64.sub
      local.set 21
      block ;; label = @2
        block ;; label = @3
          block ;; label = @4
            block ;; label = @5
              local.get 17
              i32.const 0
              i32.gt_s
              local.tee 28
              br_if 0 (;@5;)
              block ;; label = @6
                local.get 17
                br_if 0 (;@6;)
                local.get 3
                i32.const 480
                i32.add
                local.get 20
                i32.const 2
                i32.shl
                i32.add
                i32.const -4
                i32.add
                i32.load
                i32.const 23
                i32.shr_s
                local.set 29
                br 2 (;@4;)
              end
              i32.const 2
              local.set 29
              i32.const 0
              local.set 30
              local.get 21
              f64.const 0x1p-1 (;=0.5;)
              f64.ge
              i32.eqz
              br_if 3 (;@2;)
              br 2 (;@3;)
            end
            local.get 3
            i32.const 480
            i32.add
            local.get 20
            i32.const 2
            i32.shl
            i32.add
            i32.const -4
            i32.add
            local.tee 5
            local.get 5
            i32.load
            local.tee 5
            local.get 5
            local.get 11
            i32.shr_s
            local.tee 5
            local.get 11
            i32.shl
            i32.sub
            local.tee 2
            i32.store
            local.get 2
            local.get 10
            i32.shr_s
            local.set 29
            local.get 5
            local.get 27
            i32.add
            local.set 27
          end
          local.get 29
          i32.const 1
          i32.ge_s
          br_if 0 (;@3;)
          local.get 29
          local.set 30
          br 1 (;@2;)
        end
        i32.const 1
        local.set 5
        block ;; label = @3
          local.get 20
          i32.eqz
          br_if 0 (;@3;)
          block ;; label = @4
            block ;; label = @5
              block ;; label = @6
                local.get 20
                i32.const 1
                i32.ne
                br_if 0 (;@6;)
                local.get 3
                i32.const 480
                i32.add
                local.set 31
                local.get 3
                i32.load offset=480
                local.set 2
                br 1 (;@5;)
              end
              local.get 20
              i32.const 1
              i32.and
              local.set 32
              local.get 20
              i32.const 30
              i32.and
              local.set 19
              i32.const 0
              local.set 2
              i32.const 0
              local.set 30
              loop ;; label = @6
                local.get 3
                i32.const 480
                i32.add
                local.get 2
                i32.const 2
                i32.shl
                i32.add
                local.tee 5
                i32.load
                local.set 31
                i32.const 16777215
                local.set 33
                i32.const 16777215
                local.set 34
                block ;; label = @7
                  block ;; label = @8
                    block ;; label = @9
                      block ;; label = @10
                        block ;; label = @11
                          local.get 30
                          br_if 0 (;@11;)
                          i32.const 16777216
                          local.set 34
                          local.get 31
                          i32.eqz
                          br_if 1 (;@10;)
                        end
                        local.get 5
                        local.get 34
                        local.get 31
                        i32.sub
                        i32.store
                        local.get 2
                        i32.const 2
                        i32.add
                        local.set 2
                        local.get 5
                        i32.const 4
                        i32.add
                        local.set 34
                        local.get 5
                        i32.load offset=4
                        local.set 31
                        br 1 (;@9;)
                      end
                      local.get 2
                      i32.const 2
                      i32.add
                      local.set 2
                      local.get 5
                      i32.load offset=4
                      local.tee 31
                      i32.eqz
                      br_if 1 (;@8;)
                      local.get 5
                      i32.const 4
                      i32.add
                      local.set 34
                      i32.const 16777216
                      local.set 33
                    end
                    local.get 34
                    local.get 33
                    local.get 31
                    i32.sub
                    i32.store
                    i32.const 1
                    local.set 30
                    i32.const 0
                    local.set 5
                    br 1 (;@7;)
                  end
                  i32.const 0
                  local.set 30
                  i32.const 1
                  local.set 5
                end
                local.get 19
                i32.const -2
                i32.add
                local.tee 19
                br_if 0 (;@6;)
              end
              local.get 32
              i32.eqz
              br_if 2 (;@3;)
              local.get 3
              i32.const 480
              i32.add
              local.get 2
              i32.const 2
              i32.shl
              i32.add
              local.tee 31
              i32.load
              local.set 2
              local.get 30
              i32.eqz
              br_if 0 (;@5;)
              i32.const 16777215
              local.set 5
              br 1 (;@4;)
            end
            i32.const 1
            local.set 5
            local.get 2
            i32.eqz
            br_if 1 (;@3;)
            i32.const 16777216
            local.set 5
          end
          local.get 31
          local.get 5
          local.get 2
          i32.sub
          i32.store
          i32.const 0
          local.set 5
        end
        block ;; label = @3
          local.get 28
          i32.eqz
          br_if 0 (;@3;)
          i32.const 8388607
          local.set 2
          block ;; label = @4
            block ;; label = @5
              local.get 18
              br_table 1 (;@4;) 0 (;@5;) 2 (;@3;)
            end
            i32.const 4194303
            local.set 2
          end
          local.get 3
          i32.const 480
          i32.add
          local.get 20
          i32.const 2
          i32.shl
          i32.add
          i32.const -4
          i32.add
          local.tee 31
          local.get 31
          i32.load
          local.get 2
          i32.and
          i32.store
        end
        local.get 27
        i32.const 1
        i32.add
        local.set 27
        i32.const 2
        local.set 30
        local.get 29
        i32.const 2
        i32.ne
        br_if 0 (;@2;)
        f64.const 0x1p+0 (;=1;)
        local.get 21
        f64.sub
        local.tee 21
        local.get 21
        local.get 12
        f64.sub
        local.get 5
        select
        local.set 21
      end
      block ;; label = @2
        block ;; label = @3
          block ;; label = @4
            local.get 21
            f64.const 0x0p+0 (;=0;)
            f64.ne
            br_if 0 (;@4;)
            local.get 20
            i32.const -1
            i32.add
            local.tee 5
            i32.const 2
            i32.le_u
            br_if 2 (;@2;)
            i32.const 0
            local.set 31
            block ;; label = @5
              block ;; label = @6
                local.get 20
                i32.const -3
                i32.add
                local.tee 34
                i32.const 4
                i32.lt_u
                br_if 0 (;@6;)
                local.get 15
                local.get 20
                i32.const 2
                i32.shl
                i32.add
                local.set 2
                local.get 5
                local.get 34
                i32.const -4
                i32.and
                local.tee 33
                i32.sub
                local.set 5
                v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
                local.set 35
                local.get 33
                local.set 31
                loop ;; label = @7
                  local.get 2
                  v128.load align=4
                  local.get 35
                  i8x16.shuffle 12 13 14 15 8 9 10 11 4 5 6 7 0 1 2 3
                  local.get 35
                  v128.or
                  local.set 35
                  local.get 2
                  i32.const -16
                  i32.add
                  local.set 2
                  local.get 31
                  i32.const -4
                  i32.add
                  local.tee 31
                  br_if 0 (;@7;)
                end
                local.get 35
                local.get 35
                local.get 35
                i8x16.shuffle 8 9 10 11 12 13 14 15 0 1 2 3 0 1 2 3
                v128.or
                local.tee 35
                local.get 35
                local.get 35
                i8x16.shuffle 4 5 6 7 0 1 2 3 0 1 2 3 0 1 2 3
                v128.or
                i32x4.extract_lane 0
                local.set 31
                local.get 34
                local.get 33
                i32.eq
                br_if 1 (;@5;)
              end
              local.get 3
              i32.const 480
              i32.add
              local.get 5
              i32.const 2
              i32.shl
              i32.add
              local.set 2
              loop ;; label = @6
                local.get 2
                i32.load
                local.get 31
                i32.or
                local.set 31
                local.get 5
                i32.const 3
                i32.gt_u
                local.set 34
                local.get 2
                i32.const -4
                i32.add
                local.set 2
                local.get 5
                i32.const -1
                i32.add
                local.set 5
                local.get 34
                br_if 0 (;@6;)
              end
            end
            local.get 31
            i32.eqz
            br_if 2 (;@2;)
            local.get 3
            i32.const 480
            i32.add
            local.get 20
            i32.const 2
            i32.shl
            i32.add
            i32.const -4
            i32.add
            local.set 5
            loop ;; label = @5
              local.get 20
              i32.const -1
              i32.add
              local.set 20
              local.get 17
              i32.const -24
              i32.add
              local.set 17
              local.get 5
              i32.load
              local.set 2
              local.get 5
              i32.const -4
              i32.add
              local.set 5
              local.get 2
              i32.eqz
              br_if 0 (;@5;)
              br 2 (;@3;)
            end
          end
          block ;; label = @4
            local.get 21
            i32.const 1023
            local.get 17
            i32.sub
            i64.extend_i32_u
            i64.const 52
            i64.shl
            f64.reinterpret_i64
            f64.mul
            local.tee 21
            f64.const 0x1p+24 (;=16777216;)
            f64.ge
            br_if 0 (;@4;)
            local.get 3
            i32.const 480
            i32.add
            local.get 20
            i32.const 2
            i32.shl
            i32.add
            local.get 21
            i32.trunc_sat_f64_s
            i32.store
            br 1 (;@3;)
          end
          local.get 3
          i32.const 480
          i32.add
          local.get 20
          i32.const 1
          i32.add
          local.tee 5
          i32.const 2
          i32.shl
          i32.add
          local.get 21
          f64.const 0x1p-24 (;=0.00000005960464477539063;)
          f64.mul
          i32.trunc_sat_f64_s
          local.tee 2
          i32.store
          local.get 3
          i32.const 480
          i32.add
          local.get 20
          i32.const 2
          i32.shl
          i32.add
          local.get 21
          local.get 2
          f64.convert_i32_s
          f64.const -0x1p+24 (;=-16777216;)
          f64.mul
          f64.add
          i32.trunc_sat_f64_s
          i32.store
          local.get 5
          local.set 20
          local.get 9
          local.set 17
        end
        block ;; label = @3
          block ;; label = @4
            block ;; label = @5
              block ;; label = @6
                local.get 17
                i32.const 1023
                i32.gt_s
                br_if 0 (;@6;)
                local.get 17
                i32.const -1022
                i32.lt_s
                br_if 1 (;@5;)
                f64.const 0x1p+0 (;=1;)
                local.set 21
                br 3 (;@3;)
              end
              local.get 17
              i32.const 2046
              i32.gt_u
              br_if 1 (;@4;)
              local.get 17
              i32.const -1023
              i32.add
              local.set 17
              f64.const 0x1p+1023 (;=89884656743115800000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000;)
              local.set 21
              br 2 (;@3;)
            end
            block ;; label = @5
              local.get 17
              i32.const -1992
              i32.le_u
              br_if 0 (;@5;)
              local.get 17
              i32.const 969
              i32.add
              local.set 17
              f64.const 0x1p-969 (;=0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000002004168360008973;)
              local.set 21
              br 2 (;@3;)
            end
            local.get 17
            i32.const -2960
            local.get 17
            i32.const -2960
            i32.gt_u
            select
            i32.const 1938
            i32.add
            local.set 17
            f64.const 0x0p+0 (;=0;)
            local.set 21
            br 1 (;@3;)
          end
          local.get 17
          i32.const 3069
          local.get 17
          i32.const 3069
          i32.lt_u
          select
          i32.const -2046
          i32.add
          local.set 17
          f64.const inf (;=inf;)
          local.set 21
        end
        local.get 21
        local.get 17
        i32.const 1023
        i32.add
        i64.extend_i32_u
        i64.const 52
        i64.shl
        f64.reinterpret_i64
        f64.mul
        local.set 21
        block ;; label = @3
          block ;; label = @4
            local.get 20
            i32.const 1
            i32.and
            i32.eqz
            br_if 0 (;@4;)
            local.get 20
            local.set 34
            br 1 (;@3;)
          end
          local.get 3
          i32.const 320
          i32.add
          local.get 20
          i32.const 3
          i32.shl
          i32.add
          local.get 21
          local.get 3
          i32.const 480
          i32.add
          local.get 20
          i32.const 2
          i32.shl
          i32.add
          i32.load
          f64.convert_i32_s
          f64.mul
          f64.store
          local.get 21
          f64.const 0x1p-24 (;=0.00000005960464477539063;)
          f64.mul
          local.set 21
          local.get 20
          i32.const -1
          i32.add
          local.set 34
        end
        i32.const 0
        local.set 5
        i32.const 0
        local.set 31
        block ;; label = @3
          local.get 20
          i32.eqz
          br_if 0 (;@3;)
          local.get 34
          i32.const 3
          i32.shl
          local.get 3
          i32.const 320
          i32.add
          i32.add
          i32.const -8
          i32.add
          local.set 2
          local.get 34
          i32.const 2
          i32.shl
          local.get 3
          i32.const 480
          i32.add
          i32.add
          i32.const -4
          i32.add
          local.set 31
          loop ;; label = @4
            local.get 2
            local.get 21
            f64.const 0x1p-24 (;=0.00000005960464477539063;)
            f64.mul
            local.tee 22
            local.get 31
            i32.load
            f64.convert_i32_s
            f64.mul
            f64.store
            local.get 2
            i32.const 8
            i32.add
            local.get 21
            local.get 31
            i32.const 4
            i32.add
            i32.load
            f64.convert_i32_s
            f64.mul
            f64.store
            local.get 2
            i32.const -16
            i32.add
            local.set 2
            local.get 31
            i32.const -8
            i32.add
            local.set 31
            local.get 22
            f64.const 0x1p-24 (;=0.00000005960464477539063;)
            f64.mul
            local.set 21
            local.get 34
            i32.const -2
            i32.add
            local.tee 34
            i32.const -1
            i32.ne
            br_if 0 (;@4;)
          end
          local.get 20
          local.set 31
        end
        local.get 20
        i32.const 1
        i32.add
        local.set 34
        local.get 3
        i32.const 320
        i32.add
        local.get 20
        i32.const 3
        i32.shl
        i32.add
        local.set 2
        loop ;; label = @3
          local.get 2
          f64.load
          f64.const 0x1.921fb4p+0 (;=1.570796251296997;)
          f64.mul
          f64.const 0x0p+0 (;=0;)
          f64.add
          local.set 21
          block ;; label = @4
            local.get 5
            i32.eqz
            br_if 0 (;@4;)
            local.get 21
            local.get 2
            i32.const 8
            i32.add
            f64.load
            f64.const 0x1.4442dp-24 (;=0.00000007549789415861596;)
            f64.mul
            f64.add
            local.set 21
            local.get 5
            i32.const 1
            i32.eq
            br_if 0 (;@4;)
            local.get 21
            local.get 2
            i32.const 16
            i32.add
            f64.load
            f64.const 0x1.846988p-48 (;=0.000000000000005390302529957765;)
            f64.mul
            f64.add
            local.set 21
            local.get 5
            i32.const 2
            i32.eq
            br_if 0 (;@4;)
            local.get 21
            local.get 2
            i32.const 24
            i32.add
            f64.load
            f64.const 0x1.8cc516p-72 (;=0.0000000000000000000003282003415807913;)
            f64.mul
            f64.add
            local.set 21
          end
          local.get 3
          i32.const 160
          i32.add
          local.get 5
          i32.const 3
          i32.shl
          i32.add
          local.get 21
          f64.store
          local.get 5
          i32.const 1
          i32.add
          local.set 5
          local.get 2
          i32.const -8
          i32.add
          local.set 2
          local.get 31
          i32.const -1
          i32.add
          local.tee 31
          i32.const -1
          i32.ne
          br_if 0 (;@3;)
        end
        block ;; label = @3
          block ;; label = @4
            local.get 34
            i32.const 3
            i32.and
            local.tee 31
            br_if 0 (;@4;)
            f64.const 0x0p+0 (;=0;)
            local.set 21
            local.get 20
            local.set 2
            br 1 (;@3;)
          end
          local.get 3
          i32.const 160
          i32.add
          local.get 20
          i32.const 3
          i32.shl
          i32.add
          local.set 5
          f64.const 0x0p+0 (;=0;)
          local.set 21
          local.get 20
          local.set 2
          loop ;; label = @4
            local.get 2
            i32.const -1
            i32.add
            local.set 2
            local.get 21
            local.get 5
            f64.load
            f64.add
            local.set 21
            local.get 5
            i32.const -8
            i32.add
            local.set 5
            local.get 31
            i32.const -1
            i32.add
            local.tee 31
            br_if 0 (;@4;)
          end
        end
        block ;; label = @3
          local.get 20
          i32.const 3
          i32.lt_u
          br_if 0 (;@3;)
          local.get 2
          i32.const 3
          i32.shl
          local.get 3
          i32.const 160
          i32.add
          i32.add
          i32.const -24
          i32.add
          local.set 5
          loop ;; label = @4
            local.get 21
            local.get 5
            i32.const 24
            i32.add
            f64.load
            f64.add
            local.get 5
            i32.const 16
            i32.add
            f64.load
            f64.add
            local.get 5
            i32.const 8
            i32.add
            f64.load
            f64.add
            local.get 5
            f64.load
            f64.add
            local.set 21
            local.get 5
            i32.const -32
            i32.add
            local.set 5
            local.get 2
            i32.const -4
            i32.add
            local.tee 2
            i32.const -1
            i32.ne
            br_if 0 (;@4;)
          end
        end
        local.get 1
        local.get 21
        f64.neg
        local.get 21
        local.get 30
        select
        f64.store
        local.get 3
        i32.const 560
        i32.add
        global.set 0
        local.get 27
        i32.const 7
        i32.and
        return
      end
      i32.const 0
      local.set 2
      i32.const 1
      local.set 31
      local.get 14
      local.set 5
      loop ;; label = @2
        local.get 31
        local.tee 30
        i32.const 1
        i32.add
        local.set 31
        local.get 2
        i32.const 1
        i32.add
        local.set 2
        local.get 5
        i32.load
        local.set 34
        local.get 5
        i32.const -4
        i32.add
        local.set 5
        local.get 34
        i32.eqz
        br_if 0 (;@2;)
      end
      local.get 20
      local.get 2
      local.get 20
      i32.add
      local.tee 19
      i32.ge_u
      br_if 0 (;@1;)
      local.get 20
      i32.const 1
      i32.add
      local.set 27
      block ;; label = @2
        block ;; label = @3
          local.get 2
          i32.const 2
          i32.ge_u
          br_if 0 (;@3;)
          local.get 27
          local.set 31
          br 1 (;@2;)
        end
        local.get 30
        i32.const -2
        i32.and
        local.set 33
        local.get 13
        local.get 20
        i32.const 3
        i32.shl
        i32.add
        local.set 5
        local.get 6
        local.get 20
        i32.const 2
        i32.shl
        i32.add
        local.set 34
        local.get 20
        local.get 2
        i32.const -2
        i32.and
        local.tee 29
        i32.add
        local.set 20
        local.get 27
        local.get 29
        i32.add
        local.set 31
        local.get 3
        i32.const 320
        i32.add
        local.get 27
        i32.const 3
        i32.shl
        i32.add
        local.set 30
        loop ;; label = @3
          local.get 5
          local.get 34
          v128.load64_zero align=4
          f64x2.convert_low_i32x4_s
          local.tee 35
          v128.store align=8
          local.get 30
          local.get 7
          local.get 35
          f64x2.mul
          local.get 8
          f64x2.add
          v128.store align=8
          local.get 5
          i32.const 16
          i32.add
          local.set 5
          local.get 34
          i32.const 8
          i32.add
          local.set 34
          local.get 30
          i32.const 16
          i32.add
          local.set 30
          local.get 33
          i32.const -2
          i32.add
          local.tee 33
          br_if 0 (;@3;)
        end
        local.get 2
        local.get 29
        i32.eq
        br_if 1 (;@1;)
      end
      local.get 4
      local.get 31
      i32.add
      i32.const 2
      i32.shl
      i32.const 1048576
      i32.add
      local.set 5
      local.get 3
      i32.const 320
      i32.add
      local.get 31
      i32.const 3
      i32.shl
      i32.add
      local.set 2
      loop (result i32) ;; label = @2
        local.get 3
        local.get 20
        i32.const 3
        i32.shl
        i32.add
        local.get 5
        i32.load
        f64.convert_i32_s
        local.tee 21
        f64.store offset=8
        local.get 2
        local.get 0
        local.get 21
        f64.mul
        f64.const 0x0p+0 (;=0;)
        f64.add
        f64.store
        local.get 5
        i32.const 4
        i32.add
        local.set 5
        local.get 2
        i32.const 8
        i32.add
        local.set 2
        local.get 31
        local.tee 20
        i32.const 1
        i32.add
        local.set 31
        local.get 20
        local.get 19
        i32.lt_u
        br_if 0 (;@2;)
        br 1 (;@1;)
      end
    end
  )
  (func (;42;) (type 8) (param i32 i32 i32 i32 i32 i32)
    (local i32 i32 i32 i32 i32 i32 i32 f32 v128 i32 f32 f32 f32 f32 f32 i32 i32 f32 v128)
    global.get 0
    i32.const 16
    i32.sub
    local.set 6
    block ;; label = @1
      local.get 4
      local.get 5
      i32.ge_u
      br_if 0 (;@1;)
      local.get 3
      i32.eqz
      br_if 0 (;@1;)
      local.get 3
      i32.const 2
      i32.shl
      local.set 7
      local.get 3
      i32.const -4
      i32.and
      local.set 8
      local.get 3
      i32.const 3
      i32.and
      local.set 9
      local.get 1
      local.get 4
      local.get 3
      i32.mul
      i32.const 2
      i32.shl
      local.tee 10
      i32.add
      local.set 11
      local.get 0
      local.get 10
      i32.add
      local.set 12
      f32.const 0x1p+0 (;=1;)
      local.get 3
      f32.convert_i32_u
      f32.div
      local.tee 13
      f32x4.splat
      local.set 14
      local.get 3
      i32.const 4
      i32.lt_u
      local.set 15
      loop ;; label = @2
        block ;; label = @3
          block ;; label = @4
            block ;; label = @5
              local.get 15
              i32.eqz
              br_if 0 (;@5;)
              i32.const 0
              local.set 0
              f32.const -inf (;=-inf;)
              local.set 16
              br 1 (;@4;)
            end
            i32.const 0
            local.set 0
            f32.const -inf (;=-inf;)
            local.set 16
            local.get 12
            local.set 1
            loop ;; label = @5
              local.get 1
              i32.const 12
              i32.add
              f32.load
              local.tee 17
              local.get 1
              i32.const 8
              i32.add
              f32.load
              local.tee 18
              local.get 1
              i32.const 4
              i32.add
              f32.load
              local.tee 19
              local.get 1
              f32.load
              local.tee 20
              local.get 16
              local.get 20
              local.get 16
              f32.gt
              select
              local.tee 16
              local.get 19
              local.get 16
              f32.gt
              select
              local.tee 16
              local.get 18
              local.get 16
              f32.gt
              select
              local.tee 16
              local.get 17
              local.get 16
              f32.gt
              select
              local.set 16
              local.get 1
              i32.const 16
              i32.add
              local.set 1
              local.get 8
              local.get 0
              i32.const 4
              i32.add
              local.tee 0
              i32.ne
              br_if 0 (;@5;)
            end
            local.get 9
            i32.eqz
            br_if 1 (;@3;)
          end
          local.get 12
          local.get 0
          i32.const 2
          i32.shl
          i32.add
          local.set 1
          local.get 9
          local.set 0
          loop ;; label = @4
            local.get 1
            f32.load
            local.tee 17
            local.get 16
            local.get 17
            local.get 16
            f32.gt
            select
            local.set 16
            local.get 1
            i32.const 4
            i32.add
            local.set 1
            local.get 0
            i32.const -1
            i32.add
            local.tee 0
            br_if 0 (;@4;)
          end
        end
        f32.const 0x0p+0 (;=0;)
        local.set 19
        i32.const 0
        local.set 1
        local.get 3
        local.set 21
        loop ;; label = @3
          local.get 12
          local.get 1
          i32.add
          f32.load
          local.get 16
          f32.sub
          local.tee 18
          i32.reinterpret_f32
          local.tee 10
          i32.const 31
          i32.shr_u
          local.set 22
          block ;; label = @4
            block ;; label = @5
              block ;; label = @6
                block ;; label = @7
                  block ;; label = @8
                    block ;; label = @9
                      block ;; label = @10
                        local.get 10
                        i32.const 2147483647
                        i32.and
                        local.tee 0
                        i32.const 1118743632
                        i32.lt_u
                        br_if 0 (;@10;)
                        block ;; label = @11
                          local.get 0
                          i32.const 2139095040
                          i32.le_u
                          br_if 0 (;@11;)
                          local.get 18
                          local.set 17
                          br 7 (;@4;)
                        end
                        block ;; label = @11
                          local.get 0
                          i32.const 1118925335
                          i32.gt_u
                          br_if 0 (;@11;)
                          local.get 10
                          i32.const -1
                          i32.gt_s
                          br_if 2 (;@9;)
                          local.get 6
                          f32.const -0x1p-126 (;=-0.000000000000000000000000000000000000011754944;)
                          local.get 18
                          f32.div
                          f32.store offset=8
                          local.get 6
                          f32.load offset=8
                          drop
                          br 2 (;@9;)
                        end
                        block ;; label = @11
                          local.get 10
                          i32.const -1
                          i32.gt_s
                          br_if 0 (;@11;)
                          local.get 6
                          f32.const -0x1p-126 (;=-0.000000000000000000000000000000000000011754944;)
                          local.get 18
                          f32.div
                          f32.store offset=8
                          local.get 6
                          f32.load offset=8
                          drop
                          f32.const 0x0p+0 (;=0;)
                          local.set 17
                          local.get 0
                          i32.const 1120924084
                          i32.le_u
                          br_if 2 (;@9;)
                          br 7 (;@4;)
                        end
                        local.get 18
                        f32.const 0x1p+127 (;=170141180000000000000000000000000000000;)
                        f32.mul
                        local.set 17
                        br 6 (;@4;)
                      end
                      block ;; label = @10
                        local.get 0
                        i32.const 1051816472
                        i32.gt_u
                        br_if 0 (;@10;)
                        local.get 0
                        i32.const 956301312
                        i32.le_u
                        br_if 2 (;@8;)
                        i32.const 0
                        local.set 0
                        f32.const 0x0p+0 (;=0;)
                        local.set 20
                        local.get 18
                        local.set 17
                        br 5 (;@5;)
                      end
                      local.get 0
                      i32.const 1065686418
                      i32.le_u
                      br_if 2 (;@7;)
                    end
                    local.get 18
                    f32.const 0x1.715476p+0 (;=1.442695;)
                    f32.mul
                    local.get 22
                    i32.const 2
                    i32.shl
                    f32.load offset=1048840
                    f32.add
                    i32.trunc_sat_f32_s
                    local.set 0
                    br 2 (;@6;)
                  end
                  local.get 6
                  local.get 18
                  f32.const 0x1p+127 (;=170141180000000000000000000000000000000;)
                  f32.add
                  f32.store offset=12
                  local.get 18
                  f32.const 0x1p+0 (;=1;)
                  f32.add
                  local.set 17
                  local.get 6
                  f32.load offset=12
                  drop
                  br 3 (;@4;)
                end
                local.get 22
                i32.const 1
                i32.xor
                local.get 22
                i32.sub
                local.set 0
              end
              local.get 18
              local.get 0
              f32.convert_i32_s
              local.tee 17
              f32.const -0x1.62e4p-1 (;=-0.69314575;)
              f32.mul
              f32.add
              local.tee 18
              local.get 17
              f32.const 0x1.7f7d1cp-20 (;=0.0000014286068;)
              f32.mul
              local.tee 20
              f32.sub
              local.set 17
            end
            local.get 18
            local.get 17
            local.get 17
            local.get 17
            local.get 17
            f32.mul
            local.tee 23
            local.get 23
            f32.const -0x1.6aa42ap-9 (;=-0.0027667333;)
            f32.mul
            f32.const 0x1.55551ep-3 (;=0.16666625;)
            f32.add
            f32.mul
            f32.sub
            local.tee 23
            f32.mul
            f32.const 0x1p+1 (;=2;)
            local.get 23
            f32.sub
            f32.div
            local.get 20
            f32.sub
            f32.add
            f32.const 0x1p+0 (;=1;)
            f32.add
            local.set 17
            local.get 0
            i32.eqz
            br_if 0 (;@4;)
            block ;; label = @5
              block ;; label = @6
                block ;; label = @7
                  block ;; label = @8
                    local.get 0
                    i32.const 127
                    i32.gt_s
                    br_if 0 (;@8;)
                    local.get 0
                    i32.const -126
                    i32.ge_s
                    br_if 3 (;@5;)
                    local.get 17
                    f32.const 0x1p-102 (;=0.00000000000000000000000000000019721523;)
                    f32.mul
                    local.set 17
                    local.get 0
                    i32.const -229
                    i32.le_u
                    br_if 1 (;@7;)
                    local.get 0
                    i32.const 102
                    i32.add
                    local.set 0
                    br 3 (;@5;)
                  end
                  local.get 17
                  f32.const 0x1p+127 (;=170141180000000000000000000000000000000;)
                  f32.mul
                  local.set 17
                  local.get 0
                  i32.const 254
                  i32.gt_u
                  br_if 1 (;@6;)
                  local.get 0
                  i32.const -127
                  i32.add
                  local.set 0
                  br 2 (;@5;)
                end
                local.get 17
                f32.const 0x1p-102 (;=0.00000000000000000000000000000019721523;)
                f32.mul
                local.set 17
                local.get 0
                i32.const -330
                local.get 0
                i32.const -330
                i32.gt_u
                select
                i32.const 204
                i32.add
                local.set 0
                br 1 (;@5;)
              end
              local.get 17
              f32.const 0x1p+127 (;=170141180000000000000000000000000000000;)
              f32.mul
              local.set 17
              local.get 0
              i32.const 381
              local.get 0
              i32.const 381
              i32.lt_u
              select
              i32.const -254
              i32.add
              local.set 0
            end
            local.get 17
            local.get 0
            i32.const 23
            i32.shl
            i32.const 1065353216
            i32.add
            i32.const 2139095040
            i32.and
            f32.reinterpret_i32
            f32.mul
            local.set 17
          end
          local.get 11
          local.get 1
          i32.add
          local.get 17
          f32.store
          local.get 1
          i32.const 4
          i32.add
          local.set 1
          local.get 19
          local.get 17
          f32.add
          local.set 19
          local.get 21
          i32.const -1
          i32.add
          local.tee 21
          br_if 0 (;@3;)
        end
        block ;; label = @3
          block ;; label = @4
            block ;; label = @5
              local.get 19
              f32.const 0x0p+0 (;=0;)
              f32.ne
              br_if 0 (;@5;)
              i32.const 0
              local.set 1
              block ;; label = @6
                local.get 15
                br_if 0 (;@6;)
                local.get 11
                local.set 1
                local.get 8
                local.set 0
                loop ;; label = @7
                  local.get 1
                  local.get 14
                  v128.store align=4
                  local.get 1
                  i32.const 16
                  i32.add
                  local.set 1
                  local.get 0
                  i32.const -4
                  i32.add
                  local.tee 0
                  br_if 0 (;@7;)
                end
                local.get 8
                local.set 1
                local.get 3
                local.get 8
                i32.eq
                br_if 3 (;@3;)
              end
              local.get 3
              local.get 1
              i32.sub
              local.set 0
              local.get 11
              local.get 1
              i32.const 2
              i32.shl
              i32.add
              local.set 1
              br 1 (;@4;)
            end
            f32.const 0x1p+0 (;=1;)
            local.get 19
            f32.div
            local.set 16
            i32.const 0
            local.set 1
            block ;; label = @5
              local.get 15
              br_if 0 (;@5;)
              local.get 16
              f32x4.splat
              local.set 24
              local.get 11
              local.set 1
              local.get 8
              local.set 0
              loop ;; label = @6
                local.get 1
                local.get 24
                local.get 1
                v128.load align=4
                f32x4.mul
                v128.store align=4
                local.get 1
                i32.const 16
                i32.add
                local.set 1
                local.get 0
                i32.const -4
                i32.add
                local.tee 0
                br_if 0 (;@6;)
              end
              local.get 8
              local.set 1
              local.get 3
              local.get 8
              i32.eq
              br_if 2 (;@3;)
            end
            local.get 3
            local.get 1
            i32.sub
            local.set 0
            local.get 11
            local.get 1
            i32.const 2
            i32.shl
            i32.add
            local.set 1
            loop ;; label = @5
              local.get 1
              local.get 16
              local.get 1
              f32.load
              f32.mul
              f32.store
              local.get 1
              i32.const 4
              i32.add
              local.set 1
              local.get 0
              i32.const -1
              i32.add
              local.tee 0
              br_if 0 (;@5;)
              br 2 (;@3;)
            end
          end
          loop ;; label = @4
            local.get 1
            local.get 13
            f32.store
            local.get 1
            i32.const 4
            i32.add
            local.set 1
            local.get 0
            i32.const -1
            i32.add
            local.tee 0
            br_if 0 (;@4;)
          end
        end
        local.get 11
        local.get 7
        i32.add
        local.set 11
        local.get 12
        local.get 7
        i32.add
        local.set 12
        local.get 4
        i32.const 1
        i32.add
        local.tee 4
        local.get 5
        i32.ne
        br_if 0 (;@2;)
      end
    end
  )
  (func (;43;) (type 13) (param i32 i32 i32 i32 i32 i32 i32)
    (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 f32 i32 i32 v128 i32)
    block ;; label = @1
      local.get 5
      local.get 6
      i32.ge_u
      br_if 0 (;@1;)
      local.get 4
      i32.eqz
      br_if 0 (;@1;)
      local.get 4
      i32.const 2
      i32.shl
      local.set 7
      local.get 4
      i32.const 1
      i32.and
      local.set 8
      local.get 4
      i32.const -4
      i32.and
      local.set 9
      local.get 4
      i32.const -2
      i32.and
      local.set 10
      local.get 2
      local.get 5
      local.get 4
      i32.mul
      i32.const 2
      i32.shl
      local.tee 11
      i32.add
      local.set 12
      local.get 0
      local.get 11
      i32.add
      local.set 13
      local.get 1
      local.get 11
      i32.add
      local.set 14
      local.get 4
      i32.const 4
      i32.lt_u
      local.get 0
      local.get 2
      i32.sub
      i32.const -16
      i32.gt_u
      local.get 1
      local.get 2
      i32.sub
      i32.const -16
      i32.gt_u
      i32.or
      i32.or
      local.set 15
      loop ;; label = @2
        local.get 5
        local.get 4
        i32.mul
        local.set 16
        f32.const 0x0p+0 (;=0;)
        local.set 17
        i32.const 0
        local.set 18
        block ;; label = @3
          block ;; label = @4
            local.get 4
            i32.const 1
            i32.eq
            br_if 0 (;@4;)
            local.get 13
            local.set 11
            local.get 14
            local.set 19
            loop ;; label = @5
              local.get 17
              local.get 19
              f32.load
              local.get 11
              f32.load
              f32.mul
              f32.add
              local.get 19
              i32.const 4
              i32.add
              f32.load
              local.get 11
              i32.const 4
              i32.add
              f32.load
              f32.mul
              f32.add
              local.set 17
              local.get 11
              i32.const 8
              i32.add
              local.set 11
              local.get 19
              i32.const 8
              i32.add
              local.set 19
              local.get 10
              local.get 18
              i32.const 2
              i32.add
              local.tee 18
              i32.ne
              br_if 0 (;@5;)
            end
            local.get 8
            i32.eqz
            br_if 1 (;@3;)
          end
          local.get 17
          local.get 1
          local.get 18
          local.get 16
          i32.add
          i32.const 2
          i32.shl
          local.tee 11
          i32.add
          f32.load
          local.get 0
          local.get 11
          i32.add
          f32.load
          f32.mul
          f32.add
          local.set 17
        end
        i32.const 0
        local.set 11
        block ;; label = @3
          block ;; label = @4
            local.get 15
            br_if 0 (;@4;)
            local.get 17
            f32x4.splat
            local.set 20
            i32.const 0
            local.set 11
            local.get 9
            local.set 19
            loop ;; label = @5
              local.get 12
              local.get 11
              i32.add
              local.get 13
              local.get 11
              i32.add
              v128.load align=4
              local.get 14
              local.get 11
              i32.add
              v128.load align=4
              local.get 20
              f32x4.sub
              f32x4.mul
              v128.store align=4
              local.get 11
              i32.const 16
              i32.add
              local.set 11
              local.get 19
              i32.const -4
              i32.add
              local.tee 19
              br_if 0 (;@5;)
            end
            local.get 9
            local.set 11
            local.get 4
            local.get 9
            i32.eq
            br_if 1 (;@3;)
          end
          local.get 11
          i32.const 1
          i32.or
          local.set 19
          block ;; label = @4
            local.get 8
            i32.eqz
            br_if 0 (;@4;)
            local.get 2
            local.get 11
            local.get 16
            i32.add
            i32.const 2
            i32.shl
            local.tee 11
            i32.add
            local.get 0
            local.get 11
            i32.add
            f32.load
            local.get 1
            local.get 11
            i32.add
            f32.load
            local.get 17
            f32.sub
            f32.mul
            f32.store
            local.get 19
            local.set 11
          end
          local.get 4
          local.get 19
          i32.eq
          br_if 0 (;@3;)
          local.get 4
          local.get 11
          i32.sub
          local.set 19
          local.get 11
          i32.const 2
          i32.shl
          local.set 11
          loop ;; label = @4
            local.get 12
            local.get 11
            i32.add
            local.tee 18
            local.get 13
            local.get 11
            i32.add
            local.tee 16
            f32.load
            local.get 14
            local.get 11
            i32.add
            local.tee 21
            f32.load
            local.get 17
            f32.sub
            f32.mul
            f32.store
            local.get 18
            i32.const 4
            i32.add
            local.get 16
            i32.const 4
            i32.add
            f32.load
            local.get 21
            i32.const 4
            i32.add
            f32.load
            local.get 17
            f32.sub
            f32.mul
            f32.store
            local.get 11
            i32.const 8
            i32.add
            local.set 11
            local.get 19
            i32.const -2
            i32.add
            local.tee 19
            br_if 0 (;@4;)
          end
        end
        local.get 12
        local.get 7
        i32.add
        local.set 12
        local.get 13
        local.get 7
        i32.add
        local.set 13
        local.get 14
        local.get 7
        i32.add
        local.set 14
        local.get 5
        i32.const 1
        i32.add
        local.tee 5
        local.get 6
        i32.ne
        br_if 0 (;@2;)
      end
    end
  )
  (func (;44;) (type 1) (param i32 i32 i32 i32 i32)
    (local i32 i32 i32 i32 i32 i32)
    block ;; label = @1
      local.get 4
      local.get 3
      i32.le_u
      br_if 0 (;@1;)
      block ;; label = @2
        local.get 4
        local.get 3
        i32.sub
        local.tee 5
        i32.const 8
        i32.lt_u
        br_if 0 (;@2;)
        local.get 1
        local.get 2
        i32.sub
        i32.const -16
        i32.gt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 2
        i32.sub
        i32.const -16
        i32.gt_u
        br_if 0 (;@2;)
        local.get 1
        local.get 3
        i32.const 2
        i32.shl
        local.tee 6
        i32.add
        local.set 7
        local.get 0
        local.get 6
        i32.add
        local.set 8
        local.get 2
        local.get 6
        i32.add
        local.set 6
        local.get 3
        local.get 5
        i32.const -4
        i32.and
        local.tee 9
        i32.add
        local.set 3
        local.get 9
        local.set 10
        loop ;; label = @3
          local.get 6
          local.get 7
          v128.load align=4
          v128.const i32x4 0x3f000000 0x3f000000 0x3f000000 0x3f000000
          f32x4.mul
          local.get 8
          v128.load align=4
          f32x4.div
          v128.store align=4
          local.get 7
          i32.const 16
          i32.add
          local.set 7
          local.get 8
          i32.const 16
          i32.add
          local.set 8
          local.get 6
          i32.const 16
          i32.add
          local.set 6
          local.get 10
          i32.const -4
          i32.add
          local.tee 10
          br_if 0 (;@3;)
        end
        local.get 5
        local.get 9
        i32.eq
        br_if 1 (;@1;)
      end
      local.get 3
      i32.const 1
      i32.add
      local.set 7
      block ;; label = @2
        local.get 4
        local.get 3
        i32.sub
        i32.const 1
        i32.and
        i32.eqz
        br_if 0 (;@2;)
        local.get 2
        local.get 3
        i32.const 2
        i32.shl
        local.tee 8
        i32.add
        local.get 1
        local.get 8
        i32.add
        f32.load
        f32.const 0x1p-1 (;=0.5;)
        f32.mul
        local.get 0
        local.get 8
        i32.add
        f32.load
        f32.div
        f32.store
        local.get 7
        local.set 3
      end
      local.get 4
      local.get 7
      i32.eq
      br_if 0 (;@1;)
      local.get 4
      local.get 3
      i32.sub
      local.set 10
      local.get 1
      local.get 3
      i32.const 2
      i32.shl
      local.tee 6
      i32.add
      local.set 7
      local.get 0
      local.get 6
      i32.add
      local.set 8
      local.get 2
      local.get 6
      i32.add
      local.set 6
      loop ;; label = @2
        local.get 6
        local.get 7
        f32.load
        f32.const 0x1p-1 (;=0.5;)
        f32.mul
        local.get 8
        f32.load
        f32.div
        f32.store
        local.get 6
        i32.const 4
        i32.add
        local.get 7
        i32.const 4
        i32.add
        f32.load
        f32.const 0x1p-1 (;=0.5;)
        f32.mul
        local.get 8
        i32.const 4
        i32.add
        f32.load
        f32.div
        f32.store
        local.get 7
        i32.const 8
        i32.add
        local.set 7
        local.get 8
        i32.const 8
        i32.add
        local.set 8
        local.get 6
        i32.const 8
        i32.add
        local.set 6
        local.get 10
        i32.const -2
        i32.add
        local.tee 10
        br_if 0 (;@2;)
      end
    end
  )
  (func (;45;) (type 6) (param i32 i32 i32 i32)
    (local i32 i32 i32)
    block ;; label = @1
      local.get 2
      i32.const 4
      i32.add
      local.get 3
      i32.gt_u
      br_if 0 (;@1;)
      local.get 1
      local.get 2
      i32.const 2
      i32.shl
      local.tee 4
      i32.add
      local.set 5
      local.get 0
      local.get 4
      i32.add
      local.set 4
      loop ;; label = @2
        local.get 5
        local.get 4
        v128.load align=1
        f32x4.sqrt
        v128.store align=1
        local.get 5
        i32.const 16
        i32.add
        local.set 5
        local.get 4
        i32.const 16
        i32.add
        local.set 4
        local.get 2
        local.tee 6
        i32.const 4
        i32.add
        local.set 2
        local.get 6
        i32.const 8
        i32.add
        local.get 3
        i32.le_u
        br_if 0 (;@2;)
      end
    end
    block ;; label = @1
      local.get 3
      local.get 2
      i32.le_u
      br_if 0 (;@1;)
      local.get 2
      i32.const 1
      i32.add
      local.set 5
      block ;; label = @2
        local.get 3
        local.get 2
        i32.sub
        i32.const 1
        i32.and
        i32.eqz
        br_if 0 (;@2;)
        local.get 1
        local.get 2
        i32.const 2
        i32.shl
        local.tee 4
        i32.add
        local.get 0
        local.get 4
        i32.add
        v128.load32_zero
        f32x4.sqrt
        f32x4.extract_lane 0
        f32.store
        local.get 5
        local.set 2
      end
      local.get 3
      local.get 5
      i32.eq
      br_if 0 (;@1;)
      local.get 3
      local.get 2
      i32.sub
      local.set 6
      local.get 0
      local.get 2
      i32.const 2
      i32.shl
      local.tee 4
      i32.add
      local.set 5
      local.get 1
      local.get 4
      i32.add
      local.set 4
      loop ;; label = @2
        local.get 4
        local.get 5
        v128.load32_zero
        f32x4.sqrt
        f32x4.extract_lane 0
        f32.store
        local.get 4
        i32.const 4
        i32.add
        local.get 5
        i32.const 4
        i32.add
        v128.load32_zero
        f32x4.sqrt
        f32x4.extract_lane 0
        f32.store
        local.get 5
        i32.const 8
        i32.add
        local.set 5
        local.get 4
        i32.const 8
        i32.add
        local.set 4
        local.get 6
        i32.const -2
        i32.add
        local.tee 6
        br_if 0 (;@2;)
      end
    end
  )
  (func (;46;) (type 1) (param i32 i32 i32 i32 i32)
    (local i32 i32 i32 i32 i32 i32)
    block ;; label = @1
      local.get 3
      i32.const 4
      i32.add
      local.get 4
      i32.gt_u
      br_if 0 (;@1;)
      local.get 2
      local.get 3
      i32.const 2
      i32.shl
      local.tee 5
      i32.add
      local.set 6
      local.get 1
      local.get 5
      i32.add
      local.set 7
      local.get 0
      local.get 5
      i32.add
      local.set 5
      loop ;; label = @2
        local.get 6
        local.get 5
        v128.load align=1
        local.get 7
        v128.load align=1
        f32x4.sub
        v128.store align=1
        local.get 6
        i32.const 16
        i32.add
        local.set 6
        local.get 7
        i32.const 16
        i32.add
        local.set 7
        local.get 5
        i32.const 16
        i32.add
        local.set 5
        local.get 3
        local.tee 8
        i32.const 4
        i32.add
        local.set 3
        local.get 8
        i32.const 8
        i32.add
        local.get 4
        i32.le_u
        br_if 0 (;@2;)
      end
    end
    block ;; label = @1
      local.get 4
      local.get 3
      i32.le_u
      br_if 0 (;@1;)
      block ;; label = @2
        local.get 4
        local.get 3
        i32.sub
        local.tee 9
        i32.const 8
        i32.lt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 2
        i32.sub
        i32.const -16
        i32.gt_u
        br_if 0 (;@2;)
        local.get 1
        local.get 2
        i32.sub
        i32.const -16
        i32.gt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 3
        i32.const 2
        i32.shl
        local.tee 5
        i32.add
        local.set 6
        local.get 1
        local.get 5
        i32.add
        local.set 7
        local.get 2
        local.get 5
        i32.add
        local.set 5
        local.get 3
        local.get 9
        i32.const -4
        i32.and
        local.tee 10
        i32.add
        local.set 3
        local.get 10
        local.set 8
        loop ;; label = @3
          local.get 5
          local.get 6
          v128.load align=4
          local.get 7
          v128.load align=4
          f32x4.sub
          v128.store align=4
          local.get 6
          i32.const 16
          i32.add
          local.set 6
          local.get 7
          i32.const 16
          i32.add
          local.set 7
          local.get 5
          i32.const 16
          i32.add
          local.set 5
          local.get 8
          i32.const -4
          i32.add
          local.tee 8
          br_if 0 (;@3;)
        end
        local.get 9
        local.get 10
        i32.eq
        br_if 1 (;@1;)
      end
      local.get 3
      i32.const 1
      i32.add
      local.set 6
      block ;; label = @2
        local.get 4
        local.get 3
        i32.sub
        i32.const 1
        i32.and
        i32.eqz
        br_if 0 (;@2;)
        local.get 2
        local.get 3
        i32.const 2
        i32.shl
        local.tee 7
        i32.add
        local.get 0
        local.get 7
        i32.add
        f32.load
        local.get 1
        local.get 7
        i32.add
        f32.load
        f32.sub
        f32.store
        local.get 6
        local.set 3
      end
      local.get 4
      local.get 6
      i32.eq
      br_if 0 (;@1;)
      local.get 4
      local.get 3
      i32.sub
      local.set 8
      local.get 0
      local.get 3
      i32.const 2
      i32.shl
      local.tee 5
      i32.add
      local.set 6
      local.get 1
      local.get 5
      i32.add
      local.set 7
      local.get 2
      local.get 5
      i32.add
      local.set 5
      loop ;; label = @2
        local.get 5
        local.get 6
        f32.load
        local.get 7
        f32.load
        f32.sub
        f32.store
        local.get 5
        i32.const 4
        i32.add
        local.get 6
        i32.const 4
        i32.add
        f32.load
        local.get 7
        i32.const 4
        i32.add
        f32.load
        f32.sub
        f32.store
        local.get 6
        i32.const 8
        i32.add
        local.set 6
        local.get 7
        i32.const 8
        i32.add
        local.set 7
        local.get 5
        i32.const 8
        i32.add
        local.set 5
        local.get 8
        i32.const -2
        i32.add
        local.tee 8
        br_if 0 (;@2;)
      end
    end
  )
  (func (;47;) (type 8) (param i32 i32 i32 i32 i32 i32)
    (local i32 i32 i32 i32 i32 f32 i32)
    block ;; label = @1
      local.get 5
      local.get 4
      i32.le_u
      br_if 0 (;@1;)
      block ;; label = @2
        block ;; label = @3
          local.get 3
          i32.eqz
          br_if 0 (;@3;)
          local.get 2
          i32.eqz
          br_if 1 (;@2;)
          local.get 3
          local.get 2
          i32.mul
          local.set 6
          local.get 2
          i32.const -4
          i32.and
          local.set 7
          local.get 2
          i32.const 3
          i32.and
          local.set 8
          local.get 2
          i32.const 4
          i32.lt_u
          local.set 9
          loop ;; label = @4
            local.get 0
            local.get 6
            local.get 4
            local.get 3
            i32.div_u
            local.tee 2
            i32.mul
            i32.const 2
            i32.shl
            i32.add
            local.get 4
            local.get 2
            local.get 3
            i32.mul
            i32.sub
            i32.const 2
            i32.shl
            i32.add
            local.set 10
            block ;; label = @5
              block ;; label = @6
                block ;; label = @7
                  local.get 9
                  i32.eqz
                  br_if 0 (;@7;)
                  f32.const 0x0p+0 (;=0;)
                  local.set 11
                  i32.const 0
                  local.set 2
                  br 1 (;@6;)
                end
                f32.const 0x0p+0 (;=0;)
                local.set 11
                i32.const 0
                local.set 2
                loop ;; label = @7
                  local.get 11
                  local.get 10
                  local.get 2
                  local.get 3
                  i32.mul
                  i32.const 2
                  i32.shl
                  i32.add
                  f32.load
                  f32.add
                  local.get 10
                  local.get 2
                  i32.const 1
                  i32.or
                  local.get 3
                  i32.mul
                  i32.const 2
                  i32.shl
                  i32.add
                  f32.load
                  f32.add
                  local.get 10
                  local.get 2
                  i32.const 2
                  i32.or
                  local.get 3
                  i32.mul
                  i32.const 2
                  i32.shl
                  i32.add
                  f32.load
                  f32.add
                  local.get 10
                  local.get 2
                  i32.const 3
                  i32.or
                  local.get 3
                  i32.mul
                  i32.const 2
                  i32.shl
                  i32.add
                  f32.load
                  f32.add
                  local.set 11
                  local.get 7
                  local.get 2
                  i32.const 4
                  i32.add
                  local.tee 2
                  i32.ne
                  br_if 0 (;@7;)
                end
                local.get 8
                i32.eqz
                br_if 1 (;@5;)
              end
              local.get 8
              local.set 12
              loop ;; label = @6
                local.get 11
                local.get 10
                local.get 2
                local.get 3
                i32.mul
                i32.const 2
                i32.shl
                i32.add
                f32.load
                f32.add
                local.set 11
                local.get 2
                i32.const 1
                i32.add
                local.set 2
                local.get 12
                i32.const -1
                i32.add
                local.tee 12
                br_if 0 (;@6;)
              end
            end
            local.get 1
            local.get 4
            i32.const 2
            i32.shl
            i32.add
            local.get 11
            f32.store
            local.get 4
            i32.const 1
            i32.add
            local.tee 4
            local.get 5
            i32.ne
            br_if 0 (;@4;)
            br 3 (;@1;)
          end
        end
        call 3
        unreachable
      end
      local.get 5
      local.get 4
      i32.sub
      i32.const 2
      i32.shl
      local.tee 2
      i32.eqz
      br_if 0 (;@1;)
      local.get 1
      local.get 4
      i32.const 2
      i32.shl
      i32.add
      i32.const 0
      local.get 2
      memory.fill
      return
    end
  )
  (func (;48;) (type 16) (param i32 i32 i32)
    (local i32 i32 f32 i32)
    block ;; label = @1
      local.get 2
      br_if 0 (;@1;)
      local.get 1
      f32.const 0x0p+0 (;=0;)
      f32.store
      return
    end
    local.get 2
    i32.const 3
    i32.and
    local.set 3
    block ;; label = @1
      block ;; label = @2
        block ;; label = @3
          local.get 2
          i32.const 4
          i32.ge_u
          br_if 0 (;@3;)
          i32.const 0
          local.set 4
          f32.const 0x0p+0 (;=0;)
          local.set 5
          br 1 (;@2;)
        end
        local.get 2
        i32.const -4
        i32.and
        local.set 6
        i32.const 0
        local.set 4
        f32.const 0x0p+0 (;=0;)
        local.set 5
        local.get 0
        local.set 2
        loop ;; label = @3
          local.get 5
          local.get 2
          f32.load
          f32.add
          local.get 2
          i32.const 4
          i32.add
          f32.load
          f32.add
          local.get 2
          i32.const 8
          i32.add
          f32.load
          f32.add
          local.get 2
          i32.const 12
          i32.add
          f32.load
          f32.add
          local.set 5
          local.get 2
          i32.const 16
          i32.add
          local.set 2
          local.get 6
          local.get 4
          i32.const 4
          i32.add
          local.tee 4
          i32.ne
          br_if 0 (;@3;)
        end
        local.get 3
        i32.eqz
        br_if 1 (;@1;)
      end
      local.get 0
      local.get 4
      i32.const 2
      i32.shl
      i32.add
      local.set 2
      loop ;; label = @2
        local.get 5
        local.get 2
        f32.load
        f32.add
        local.set 5
        local.get 2
        i32.const 4
        i32.add
        local.set 2
        local.get 3
        i32.const -1
        i32.add
        local.tee 3
        br_if 0 (;@2;)
      end
    end
    local.get 1
    local.get 5
    f32.store
  )
  (func (;49;) (type 1) (param i32 i32 i32 i32 i32)
    (local v128 i32 i32 f32 i32)
    block ;; label = @1
      block ;; label = @2
        local.get 3
        i32.const 4
        i32.add
        local.get 4
        i32.le_u
        br_if 0 (;@2;)
        v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
        local.set 5
        br 1 (;@1;)
      end
      local.get 0
      local.get 3
      i32.const 2
      i32.shl
      i32.add
      local.set 6
      v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000
      local.set 5
      loop ;; label = @2
        local.get 3
        local.tee 7
        i32.const 4
        i32.add
        local.set 3
        local.get 5
        local.get 6
        v128.load align=1
        f32x4.add
        local.set 5
        local.get 6
        i32.const 16
        i32.add
        local.set 6
        local.get 7
        i32.const 8
        i32.add
        local.get 4
        i32.le_u
        br_if 0 (;@2;)
      end
    end
    local.get 5
    f32x4.extract_lane 0
    local.get 5
    f32x4.extract_lane 1
    f32.add
    local.get 5
    f32x4.extract_lane 2
    f32.add
    local.get 5
    f32x4.extract_lane 3
    f32.add
    local.set 8
    block ;; label = @1
      local.get 3
      local.get 4
      i32.ge_u
      br_if 0 (;@1;)
      block ;; label = @2
        block ;; label = @3
          local.get 4
          local.get 3
          i32.sub
          i32.const 3
          i32.and
          local.tee 9
          br_if 0 (;@3;)
          local.get 3
          local.set 7
          br 1 (;@2;)
        end
        local.get 0
        local.get 3
        i32.const 2
        i32.shl
        i32.add
        local.set 6
        local.get 3
        local.set 7
        loop ;; label = @3
          local.get 7
          i32.const 1
          i32.add
          local.set 7
          local.get 8
          local.get 6
          f32.load
          f32.add
          local.set 8
          local.get 6
          i32.const 4
          i32.add
          local.set 6
          local.get 9
          i32.const -1
          i32.add
          local.tee 9
          br_if 0 (;@3;)
        end
      end
      local.get 3
      local.get 4
      i32.sub
      i32.const -4
      i32.gt_u
      br_if 0 (;@1;)
      local.get 4
      local.get 7
      i32.sub
      local.set 3
      local.get 0
      local.get 7
      i32.const 2
      i32.shl
      i32.add
      local.set 6
      loop ;; label = @2
        local.get 8
        local.get 6
        f32.load
        f32.add
        local.get 6
        i32.const 4
        i32.add
        f32.load
        f32.add
        local.get 6
        i32.const 8
        i32.add
        f32.load
        f32.add
        local.get 6
        i32.const 12
        i32.add
        f32.load
        f32.add
        local.set 8
        local.get 6
        i32.const 16
        i32.add
        local.set 6
        local.get 3
        i32.const -4
        i32.add
        local.tee 3
        br_if 0 (;@2;)
      end
    end
    local.get 1
    local.get 2
    i32.const 2
    i32.shl
    i32.add
    local.get 8
    f32.store
  )
  (func (;50;) (type 6) (param i32 i32 i32 i32)
    (local i32 i32 f32 f32)
    global.get 0
    i32.const 16
    i32.sub
    local.tee 4
    global.set 0
    block ;; label = @1
      local.get 3
      local.get 2
      i32.le_u
      br_if 0 (;@1;)
      local.get 3
      local.get 2
      i32.sub
      local.set 5
      local.get 0
      local.get 2
      i32.const 2
      i32.shl
      local.tee 3
      i32.add
      local.set 2
      local.get 1
      local.get 3
      i32.add
      local.set 3
      loop ;; label = @2
        block ;; label = @3
          block ;; label = @4
            block ;; label = @5
              local.get 2
              f32.load
              local.tee 6
              f32.abs
              local.tee 7
              i32.reinterpret_f32
              local.tee 1
              i32.const 1057791828
              i32.gt_u
              br_if 0 (;@5;)
              local.get 1
              i32.const 1048757624
              i32.gt_u
              br_if 1 (;@4;)
              block ;; label = @6
                local.get 1
                i32.const 8388607
                i32.gt_u
                br_if 0 (;@6;)
                local.get 4
                local.get 6
                local.get 6
                f32.mul
                f32.store offset=12
                local.get 4
                f32.load offset=12
                drop
                br 3 (;@3;)
              end
              local.get 7
              f32.const -0x1p+1 (;=-2;)
              f32.mul
              call 20
              local.tee 7
              f32.neg
              local.get 7
              f32.const 0x1p+1 (;=2;)
              f32.add
              f32.div
              local.set 7
              br 2 (;@3;)
            end
            block ;; label = @5
              local.get 1
              i32.const 1092616192
              i32.gt_u
              br_if 0 (;@5;)
              f32.const 0x1p+0 (;=1;)
              f32.const 0x1p+1 (;=2;)
              local.get 7
              local.get 7
              f32.add
              call 20
              f32.const 0x1p+1 (;=2;)
              f32.add
              f32.div
              f32.sub
              local.set 7
              br 2 (;@3;)
            end
            f32.const 0x0p+0 (;=0;)
            local.get 7
            f32.div
            f32.const 0x1p+0 (;=1;)
            f32.add
            local.set 7
            br 1 (;@3;)
          end
          local.get 7
          local.get 7
          f32.add
          call 20
          local.tee 7
          local.get 7
          f32.const 0x1p+1 (;=2;)
          f32.add
          f32.div
          local.set 7
        end
        local.get 3
        local.get 7
        f32.neg
        local.get 7
        local.get 6
        i32.reinterpret_f32
        i32.const 0
        i32.lt_s
        select
        f32.store
        local.get 2
        i32.const 4
        i32.add
        local.set 2
        local.get 3
        i32.const 4
        i32.add
        local.set 3
        local.get 5
        i32.const -1
        i32.add
        local.tee 5
        br_if 0 (;@2;)
      end
    end
    local.get 4
    i32.const 16
    i32.add
    global.set 0
  )
  (func (;51;) (type 1) (param i32 i32 i32 i32 i32)
    (local i32 i32 i32 i32 i32 i32 v128 f32)
    block ;; label = @1
      local.get 4
      local.get 3
      i32.le_u
      br_if 0 (;@1;)
      block ;; label = @2
        local.get 4
        local.get 3
        i32.sub
        local.tee 5
        i32.const 8
        i32.lt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 2
        i32.sub
        i32.const -16
        i32.gt_u
        br_if 0 (;@2;)
        local.get 1
        local.get 2
        i32.sub
        i32.const -16
        i32.gt_u
        br_if 0 (;@2;)
        local.get 0
        local.get 3
        i32.const 2
        i32.shl
        local.tee 6
        i32.add
        local.set 7
        local.get 1
        local.get 6
        i32.add
        local.set 8
        local.get 2
        local.get 6
        i32.add
        local.set 6
        local.get 3
        local.get 5
        i32.const -4
        i32.and
        local.tee 9
        i32.add
        local.set 3
        local.get 9
        local.set 10
        loop ;; label = @3
          local.get 6
          local.get 8
          v128.load align=4
          v128.const i32x4 0x3f800000 0x3f800000 0x3f800000 0x3f800000
          local.get 7
          v128.load align=4
          local.tee 11
          local.get 11
          f32x4.mul
          f32x4.sub
          f32x4.mul
          v128.store align=4
          local.get 7
          i32.const 16
          i32.add
          local.set 7
          local.get 8
          i32.const 16
          i32.add
          local.set 8
          local.get 6
          i32.const 16
          i32.add
          local.set 6
          local.get 10
          i32.const -4
          i32.add
          local.tee 10
          br_if 0 (;@3;)
        end
        local.get 5
        local.get 9
        i32.eq
        br_if 1 (;@1;)
      end
      local.get 3
      i32.const 1
      i32.add
      local.set 7
      block ;; label = @2
        local.get 4
        local.get 3
        i32.sub
        i32.const 1
        i32.and
        i32.eqz
        br_if 0 (;@2;)
        local.get 2
        local.get 3
        i32.const 2
        i32.shl
        local.tee 8
        i32.add
        local.get 1
        local.get 8
        i32.add
        f32.load
        f32.const 0x1p+0 (;=1;)
        local.get 0
        local.get 8
        i32.add
        f32.load
        local.tee 12
        local.get 12
        f32.mul
        f32.sub
        f32.mul
        f32.store
        local.get 7
        local.set 3
      end
      local.get 4
      local.get 7
      i32.eq
      br_if 0 (;@1;)
      local.get 4
      local.get 3
      i32.sub
      local.set 10
      local.get 0
      local.get 3
      i32.const 2
      i32.shl
      local.tee 6
      i32.add
      local.set 7
      local.get 1
      local.get 6
      i32.add
      local.set 8
      local.get 2
      local.get 6
      i32.add
      local.set 6
      loop ;; label = @2
        local.get 6
        local.get 8
        f32.load
        f32.const 0x1p+0 (;=1;)
        local.get 7
        f32.load
        local.tee 12
        local.get 12
        f32.mul
        f32.sub
        f32.mul
        f32.store
        local.get 6
        i32.const 4
        i32.add
        local.get 8
        i32.const 4
        i32.add
        f32.load
        f32.const 0x1p+0 (;=1;)
        local.get 7
        i32.const 4
        i32.add
        f32.load
        local.tee 12
        local.get 12
        f32.mul
        f32.sub
        f32.mul
        f32.store
        local.get 7
        i32.const 8
        i32.add
        local.set 7
        local.get 8
        i32.const 8
        i32.add
        local.set 8
        local.get 6
        i32.const 8
        i32.add
        local.set 6
        local.get 10
        i32.const -2
        i32.add
        local.tee 10
        br_if 0 (;@2;)
      end
    end
  )
  (func (;52;) (type 8) (param i32 i32 i32 i32 i32 i32)
    (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
    block ;; label = @1
      local.get 4
      local.get 5
      i32.ge_u
      br_if 0 (;@1;)
      local.get 2
      i32.eqz
      br_if 0 (;@1;)
      local.get 3
      i32.const 2
      i32.shl
      local.set 6
      local.get 3
      i32.const 4
      i32.shl
      local.set 7
      local.get 2
      i32.const 2
      i32.shl
      local.set 8
      local.get 2
      i32.const -536870916
      i32.and
      local.set 9
      local.get 2
      i32.const 3
      i32.and
      local.tee 10
      i32.const 2
      i32.add
      local.set 11
      local.get 10
      i32.const 1
      i32.add
      local.set 12
      local.get 10
      i32.const 3
      i32.add
      local.set 13
      local.get 2
      i32.const 7
      i32.gt_u
      local.get 3
      i32.const 1
      i32.eq
      i32.and
      i32.const 1
      i32.xor
      local.get 1
      local.get 4
      local.get 2
      i32.mul
      i32.const 2
      i32.shl
      i32.add
      local.tee 14
      local.get 0
      local.get 5
      local.get 2
      i32.add
      i32.const 2
      i32.shl
      i32.add
      i32.const -4
      i32.add
      i32.lt_u
      local.get 0
      local.get 4
      i32.const 2
      i32.shl
      i32.add
      local.tee 15
      local.get 1
      local.get 5
      local.get 2
      i32.mul
      i32.const 2
      i32.shl
      i32.add
      i32.lt_u
      i32.and
      local.get 2
      i32.const 536870912
      i32.and
      i32.const 29
      i32.shr_u
      i32.or
      i32.or
      local.set 16
      loop ;; label = @2
        i32.const 0
        local.set 17
        block ;; label = @3
          block ;; label = @4
            local.get 16
            br_if 0 (;@4;)
            local.get 15
            local.set 3
            local.get 14
            local.set 1
            local.get 9
            local.set 0
            loop ;; label = @5
              local.get 1
              local.get 3
              v128.load align=4
              v128.store align=4
              local.get 3
              i32.const 16
              i32.add
              local.set 3
              local.get 1
              i32.const 16
              i32.add
              local.set 1
              local.get 0
              i32.const -4
              i32.add
              local.tee 0
              br_if 0 (;@5;)
            end
            local.get 9
            local.set 17
            local.get 2
            local.get 9
            i32.eq
            br_if 1 (;@3;)
          end
          local.get 17
          local.set 18
          block ;; label = @4
            local.get 10
            i32.eqz
            br_if 0 (;@4;)
            local.get 10
            local.get 17
            i32.add
            local.set 18
            local.get 15
            local.get 6
            local.get 17
            i32.mul
            i32.add
            local.set 3
            local.get 14
            local.get 17
            i32.const 2
            i32.shl
            i32.add
            local.set 1
            local.get 10
            local.set 0
            loop ;; label = @5
              local.get 1
              local.get 3
              f32.load
              f32.store
              local.get 3
              local.get 6
              i32.add
              local.set 3
              local.get 1
              i32.const 4
              i32.add
              local.set 1
              local.get 0
              i32.const -1
              i32.add
              local.tee 0
              br_if 0 (;@5;)
            end
          end
          local.get 17
          local.get 2
          i32.sub
          i32.const -4
          i32.gt_u
          br_if 0 (;@3;)
          local.get 6
          local.get 18
          i32.mul
          local.set 19
          local.get 2
          local.get 18
          i32.sub
          local.set 0
          local.get 6
          local.get 11
          local.get 17
          i32.add
          i32.mul
          local.set 20
          local.get 6
          local.get 12
          local.get 17
          i32.add
          i32.mul
          local.set 21
          local.get 6
          local.get 13
          local.get 17
          i32.add
          i32.mul
          local.set 17
          local.get 14
          local.get 18
          i32.const 2
          i32.shl
          i32.add
          local.set 3
          local.get 15
          local.set 1
          loop ;; label = @4
            local.get 3
            local.get 1
            local.get 19
            i32.add
            f32.load
            f32.store
            local.get 3
            i32.const 4
            i32.add
            local.get 1
            local.get 21
            i32.add
            f32.load
            f32.store
            local.get 3
            i32.const 8
            i32.add
            local.get 1
            local.get 20
            i32.add
            f32.load
            f32.store
            local.get 3
            i32.const 12
            i32.add
            local.get 1
            local.get 17
            i32.add
            f32.load
            f32.store
            local.get 1
            local.get 7
            i32.add
            local.set 1
            local.get 3
            i32.const 16
            i32.add
            local.set 3
            local.get 0
            i32.const -4
            i32.add
            local.tee 0
            br_if 0 (;@4;)
          end
        end
        local.get 15
        i32.const 4
        i32.add
        local.set 15
        local.get 14
        local.get 8
        i32.add
        local.set 14
        local.get 4
        i32.const 1
        i32.add
        local.tee 4
        local.get 5
        i32.ne
        br_if 0 (;@2;)
      end
    end
  )w
  (func (;53;) (type 0)
    loop ;; label = @1
      br 0 (;@1;)
    end
  )
  (data (;0;) "\83\f9\a2\00DNn\00\fc)\15\00\d1W'\00\dd4\f5\00b\db\c0\00<\99\95\00A\90C\00cQ\fe\00\bb\de\ab\00\b7a\c5\00:n$\00\d2MB\00I\06\e0\00\09\ea.\00\1c\92\d1\00\eb\1d\fe\00)\b1\1c\00\e8>\a7\00\f55\82\00D\bb.\00\9c\e9\84\00\b4&p\00A~_\00\d6\919\00S\839\00\9c\f49\00\8b_\84\00(\f9\bd\00\f8\1f;\00\de\ff\97\00\0f\98\05\00\11/\ef\00\0aZ\8b\00m\1fm\00\cf~6\00\09\cb'\00FO\b7\00\9ef?\00-\ea_\00\ba'u\00\e5\eb\c7\00={\f1\00\f79\07\00\92R\8a\00\fbk\ea\00\1f\b1_\00\08]\8d\000\03V\00{\fcF\00\f0\abk\00 \bc\cf\006\f4\9a\00\e3\a9\1d\00^a\91\00\08\1b\e6\00\85\99e\00\a0\14_\00\8d@h\00\80\d8\ff\00'sM\00\06\061\00\caV\15\00\c9\a8s\00{\e2`\00k\8c\c0\00\00\00\00?\00\00\00\bfQ\b4\f0\b2\96\b1D\b0\f9\ae\b6\ady\acC\ab\14\aa\eb\a8\c8\a7\aa\a6\92\a5\80\a4s\a3k\a2h\a1j\a0p\9f{\9e\8a\9d\9d\9c\b5\9b\d1\9a\f0\99\13\99:\98e\97\93\96\c4\95\f8\940\94k\93\a9\92\ea\91.\91u\90\be\8f\0a\8fY\8e\aa\8d\fe\8cT\8c\ac\8b\07\8bd\8a\c4\89%\89\89\88\ee\87V\87\c0\86+\86\99\85\08\85y\84\ec\83a\83\d8\82P\82\c9\81E\81\c2\80@\80\02\ff\0e\fd%\fbG\f9s\f7\aa\f5\ea\f34\f2\87\f0\e3\eeG\ed\b3\eb'\ea\a3\e8'\e7\b2\e5C\e4\dc\e2z\e1 \e0\cb\de}\dd4\dc\f1\da\b3\d9{\d8H\d7\1a\d6\f1\d4\cd\d3\ad\d2\92\d1{\d0i\cf[\ceQ\cdJ\ccH\cbJ\caO\c9X\c8d\c7t\c6\87\c5\9d\c4\b7\c3\d4\c2\f4\c1\16\c1<\c0e\bf\90\be\be\bd\ef\bc#\bcY\bb\91\ba\cc\b9\0a\b9J\b8\8c\b7\d0\b6\17\b6`\b5")
)
