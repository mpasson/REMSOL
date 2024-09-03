fn main() {
    println!("Enter a string: ");
    let mut input = String::new();
    std::io::stdin()
        .read_line(&mut input)
        .expect("Failed to read line");
    let first = first_word(&input);
    println!("The first word fo '{}' is: {}", input, first);
}

fn first_word(s: &String) -> String {
    let mut new_string = String::new();
    for i in s.chars() {
        if i != ' ' {
            new_string.push(i);
        } else {
            break;
        }
    }
    new_string
}
