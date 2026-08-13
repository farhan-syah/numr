//! Macro-based code generation for var_* functions
//!
//! These macros eliminate boilerplate by generating var_* functions with
//! consistent patterns. Each macro handles a specific backward constructor
//! signature pattern.

/// Binary operation with shape-based backward (add, sub)
macro_rules! impl_var_binary_op_shapes {
    ($(#[$meta:meta])* $fn_name:ident, $op_method:ident, $backward_ty:ident) => {
        $(#[$meta])*
        pub fn $fn_name<R, C>(a: &Var<R>, b: &Var<R>, client: &C) -> Result<Var<R>>
        where
            R: Runtime,
            C: RuntimeClient<R> + TensorOps<R>,
            R::Client: TensorOps<R>,
        {
            let output = client.$op_method(a.tensor(), b.tensor())?;

            if a.requires_grad() || b.requires_grad() {
                let grad_fn = $backward_ty::<R>::new(
                    a.id(),
                    b.id(),
                    a.shape(),
                    b.shape(),
                    a.grad_fn().cloned(),
                    b.grad_fn().cloned(),
                );
                Ok(Var::from_op(output, std::sync::Arc::new(grad_fn)))
            } else {
                Ok(Var::new(output, false))
            }
        }
    };
}

/// Binary operation with tensor-based backward (mul, div, matmul)
macro_rules! impl_var_binary_op_tensors {
    ($(#[$meta:meta])* $fn_name:ident, $op_method:ident, $backward_ty:ident) => {
        $(#[$meta])*
        pub fn $fn_name<R, C>(a: &Var<R>, b: &Var<R>, client: &C) -> Result<Var<R>>
        where
            R: Runtime,
            C: RuntimeClient<R> + TensorOps<R>,
            R::Client: TensorOps<R>,
        {
            let output = client.$op_method(a.tensor(), b.tensor())?;

            if a.requires_grad() || b.requires_grad() {
                let grad_fn = $backward_ty::<R>::new(
                    a.id(),
                    b.id(),
                    a.tensor().clone(),
                    b.tensor().clone(),
                    a.grad_fn().cloned(),
                    b.grad_fn().cloned(),
                );
                Ok(Var::from_op(output, std::sync::Arc::new(grad_fn)))
            } else {
                Ok(Var::new(output, false))
            }
        }
    };
}

/// Unary operation with id-only backward (neg)
macro_rules! impl_var_unary_op_id {
    ($(#[$meta:meta])* $fn_name:ident, $op_method:ident, $backward_ty:ident) => {
        $(#[$meta])*
        pub fn $fn_name<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
        where
            R: Runtime,
            C: RuntimeClient<R> + TensorOps<R>,
            R::Client: TensorOps<R>,
        {
            let output = client.$op_method(a.tensor())?;

            if a.requires_grad() {
                let grad_fn = $backward_ty::<R>::new(a.id(), a.grad_fn().cloned());
                Ok(Var::from_op(output, std::sync::Arc::new(grad_fn)))
            } else {
                Ok(Var::new(output, false))
            }
        }
    };
}

/// Unary operation with output-based backward (exp, sqrt, tanh, recip, sigmoid)
macro_rules! impl_var_unary_op_output {
    ($(#[$meta:meta])* $fn_name:ident, $op_method:ident, $backward_ty:ident) => {
        $(#[$meta])*
        pub fn $fn_name<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
        where
            R: Runtime,
            C: RuntimeClient<R> + TensorOps<R>,
            R::Client: TensorOps<R>,
        {
            let output = client.$op_method(a.tensor())?;

            if a.requires_grad() {
                let grad_fn = $backward_ty::<R>::new(a.id(), output.clone(), a.grad_fn().cloned());
                Ok(Var::from_op(output, std::sync::Arc::new(grad_fn)))
            } else {
                Ok(Var::new(output, false))
            }
        }
    };
}

/// Unary operation with input tensor backward (log, sin, cos, square, abs, tan, relu)
macro_rules! impl_var_unary_op_input {
    ($(#[$meta:meta])* $fn_name:ident, $op_method:ident, $backward_ty:ident) => {
        $(#[$meta])*
        pub fn $fn_name<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
        where
            R: Runtime,
            C: RuntimeClient<R> + TensorOps<R>,
            R::Client: TensorOps<R>,
        {
            let output = client.$op_method(a.tensor())?;

            if a.requires_grad() {
                let grad_fn =
                    $backward_ty::<R>::new(a.id(), a.tensor().clone(), a.grad_fn().cloned());
                Ok(Var::from_op(output, std::sync::Arc::new(grad_fn)))
            } else {
                Ok(Var::new(output, false))
            }
        }
    };
}

/// Scalar operation with id-only backward (add_scalar, sub_scalar)
macro_rules! impl_var_scalar_op_id {
    ($(#[$meta:meta])* $fn_name:ident, $op_method:ident, $backward_ty:ident) => {
        $(#[$meta])*
        pub fn $fn_name<R, C>(a: &Var<R>, scalar: f64, client: &C) -> Result<Var<R>>
        where
            R: Runtime,
            C: RuntimeClient<R> + ScalarOps<R>,
            R::Client: ScalarOps<R>,
        {
            let output = client.$op_method(a.tensor(), scalar)?;

            if a.requires_grad() {
                let grad_fn = $backward_ty::<R>::new(a.id(), a.grad_fn().cloned());
                Ok(Var::from_op(output, std::sync::Arc::new(grad_fn)))
            } else {
                Ok(Var::new(output, false))
            }
        }
    };
}

/// Scalar operation with scalar value backward (mul_scalar, div_scalar)
macro_rules! impl_var_scalar_op_scalar {
    ($(#[$meta:meta])* $fn_name:ident, $op_method:ident, $backward_ty:ident) => {
        $(#[$meta])*
        pub fn $fn_name<R, C>(a: &Var<R>, scalar: f64, client: &C) -> Result<Var<R>>
        where
            R: Runtime,
            C: RuntimeClient<R> + ScalarOps<R>,
            R::Client: ScalarOps<R>,
        {
            let output = client.$op_method(a.tensor(), scalar)?;

            if a.requires_grad() {
                let grad_fn = $backward_ty::<R>::new(a.id(), scalar, a.grad_fn().cloned());
                Ok(Var::from_op(output, std::sync::Arc::new(grad_fn)))
            } else {
                Ok(Var::new(output, false))
            }
        }
    };
}

/// Unary operation with output-based backward that requires ScalarOps (sqrt, tanh, recip, sigmoid)
///
/// These operations save the output tensor and require ScalarOps for backward computation.
macro_rules! impl_var_unary_op_output_scalar {
    ($(#[$meta:meta])* $fn_name:ident, $op_method:ident, $backward_ty:ident) => {
        $(#[$meta])*
        pub fn $fn_name<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
        where
            R: Runtime<DType = crate::dtype::DType>,
            C: RuntimeClient<R> + TensorOps<R>,
            R::Client: TensorOps<R> + ScalarOps<R>,
        {
            let output = client.$op_method(a.tensor())?;

            if a.requires_grad() {
                let grad_fn = $backward_ty::<R>::new(a.id(), output.clone(), a.grad_fn().cloned());
                Ok(Var::from_op(output, std::sync::Arc::new(grad_fn)))
            } else {
                Ok(Var::new(output, false))
            }
        }
    };
}

/// Unary operation with input-based backward that requires ScalarOps (square, tan)
///
/// These operations save the input tensor and require ScalarOps for backward computation.
macro_rules! impl_var_unary_op_input_scalar {
    ($(#[$meta:meta])* $fn_name:ident, $op_method:ident, $backward_ty:ident) => {
        $(#[$meta])*
        pub fn $fn_name<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
        where
            R: Runtime,
            C: RuntimeClient<R> + TensorOps<R>,
            R::Client: TensorOps<R> + ScalarOps<R>,
        {
            let output = client.$op_method(a.tensor())?;

            if a.requires_grad() {
                let grad_fn =
                    $backward_ty::<R>::new(a.id(), a.tensor().clone(), a.grad_fn().cloned());
                Ok(Var::from_op(output, std::sync::Arc::new(grad_fn)))
            } else {
                Ok(Var::new(output, false))
            }
        }
    };
}

pub(crate) use impl_var_binary_op_shapes;
pub(crate) use impl_var_binary_op_tensors;
pub(crate) use impl_var_scalar_op_id;
pub(crate) use impl_var_scalar_op_scalar;
pub(crate) use impl_var_unary_op_id;
pub(crate) use impl_var_unary_op_input;
pub(crate) use impl_var_unary_op_input_scalar;
pub(crate) use impl_var_unary_op_output;
pub(crate) use impl_var_unary_op_output_scalar;
