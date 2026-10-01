(* A short tour. Run:  python -m hm infer examples/tour.ml
                       python -m hm run   examples/tour.ml *)

(* Let polymorphism: pair is used at two different types below. *)
let pair x y = (x, y)
let swap p = match p with (a, b) -> (b, a)

(* Higher order functions get fully general types without annotations. *)
let twice f x = f (f x)
let rec compose_all fs =
  match fs with
  | [] -> fun x -> x
  | f :: rest -> fun x -> f (compose_all rest x)

let rec insert x l =
  match l with
  | [] -> [x]
  | y :: ys -> if x <= y then x :: l else y :: insert x ys

let sort l = fold_right insert l []

let rec unique l =
  match l with
  | a :: b :: rest -> if a = b then unique (b :: rest) else a :: unique (b :: rest)
  | short -> short

let rec take n l =
  match n, l with
  | 0, _ -> []
  | _, [] -> []
  | n, x :: xs -> x :: take (n - 1) xs

let sum = fold_left (fun a b -> a + b) 0
let squares n = map (fun i -> i * i) (range 1 (n + 1))

let show_list show l =
  "[" ^ fold_left (fun acc x -> if acc = "" then show x else acc ^ "; " ^ show x) "" l ^ "]"

let () =
  print_endline (show_list string_of_int (sort [5; 3; 9; 1; 4]));
  print_endline (show_list string_of_int (twice (map (fun x -> x + 1)) [1; 2; 3]));
  print_endline (string_of_int (sum (squares 10)));
  print_endline (show_list string_of_bool (map fst [pair true 1; pair false 2]))

;; swap (1, "one");;
;; compose_all [(fun x -> x * 2); (fun x -> x + 3)] 10;;
;; take 3 (squares 100);;
