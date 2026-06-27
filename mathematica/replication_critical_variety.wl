(* ::Package:: *)

(* Mathematica/Wolfram Language workflow for the self-replication example (Section 6.1). *)
(* Self-replication model of Gijima--Peacock-Lopez.                                       *)
(* Species order: x1=A, x2=B, x3=P, x4=Ia, x5=Ib, x6=I.                                   *)
(* Reactions (all reversible):                                                            *)
(*   X1 + X3 <-> X4   (k1, km1)                                                            *)
(*   X2 + X4 <-> X6   (k2, km2)                                                            *)
(*   X2 + X3 <-> X5   (k3, km3)                                                            *)
(*   X1 + X5 <-> X6   (k4, km4)                                                            *)
(*   X6     <-> 2 X3  (k5, km5)                                                            *)

ClearAll["Global`*"];

xvars = {x1, x2, x3, x4, x5, x6};
kvars = {k1, k2, k3, k4, k5, km1, km2, km3, km4, km5};  (* paper column order *)
pqr   = {p, q, r};                                      (* lumping-ansatz parameters *)

(* Constrained lumping ansatz of Section 6.1.                          *)
(*   Row 1: y1 = x3 + p x4 + q x5 + r x6  (the lumping ansatz);         *)
(*   Rows 2, 3: prescribed stoichiometric first integrals.             *)
T = {{0, 0, 1, p, q, r},
     {1, -1, 0, 1, -1, 0},
     {0, 1, 1, 1, 2, 2}};

(* Right kernel basis of T, parametrised by the free species x4, x5, x6. *)
B = {{p - 2, q - 1, r - 2},
     {p - 1, q - 2, r - 2},
     {-p,    -q,    -r   },
     {1,     0,     0    },
     {0,     1,     0    },
     {0,     0,     1    }};

kernelCheck = Simplify[T . B];
If[kernelCheck =!= ConstantArray[0, {3, 3}],
   Print["ERROR: B is not a right kernel basis for T: ", kernelCheck]];

(* Mass-action vector field for the self-replication network. *)
x1d = -k1*x1*x3 + km1*x4 - k4*x1*x5 + km4*x6;
x2d = -k2*x2*x4 + km2*x6 - k3*x2*x3 + km3*x5;
x3d = -k1*x1*x3 + km1*x4 - k3*x2*x3 + km3*x5 + 2*k5*x6 - 2*km5*x3^2;
x4d =  k1*x1*x3 - km1*x4 - k2*x2*x4 + km2*x6;
x5d =  k3*x2*x3 - km3*x5 - k4*x1*x5 + km4*x6;
x6d =  k2*x2*x4 - km2*x6 + k4*x1*x5 - km4*x6 - k5*x6 + km5*x3^2;

F = {x1d, x2d, x3d, x4d, x5d, x6d};

(* Conservation checks: rows 2 and 3 of T are stoichiometric first integrals, *)
(* so their time derivatives must be identically zero for all parameters.     *)
firstIntegral2Check = Factor[T[[2]] . F];
firstIntegral3Check = Factor[T[[3]] . F];
If[firstIntegral2Check =!= 0, Print["ERROR: T row 2 is not a first integral: ", firstIntegral2Check]];
If[firstIntegral3Check =!= 0, Print["ERROR: T row 3 is not a first integral: ", firstIntegral3Check]];

jac = D[F, {xvars}];
U = Together[T . jac . B];

coefficientsInStateVariables[expr_] := Values[CoefficientRules[Expand[expr], xvars]];
conditions = Factor[DeleteCases[Flatten[coefficientsInStateVariables /@ Flatten[U]], 0]];

(* The conditions are homogeneous-linear in the rate constants.  Assembling   *)
(* the coefficient matrix in kvars recovers the matrix reported in Section 6.1.*)
arrays = CoefficientArrays[conditions, kvars];
constantPart = Normal[arrays[[1]]];
M = Normal[arrays[[2]]];

If[Factor[constantPart] =!= ConstantArray[0, Length[conditions]],
   Print["ERROR: conditions are not homogeneous-linear in rate constants. Constant part: ", constantPart]];

Print["T.B = "]; Print[MatrixForm[kernelCheck]];
Print["First-integral check (T row 2) = ", firstIntegral2Check];
Print["First-integral check (T row 3) = ", firstIntegral3Check];
Print["Number of coefficient conditions = ", Length[conditions]];
Print["Coefficient matrix M, rate-constant order ", kvars, ":"];
Print[MatrixForm[M]];
Print["Coefficient matrix M has dimensions ", Dimensions[M]];
