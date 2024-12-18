use proc_macro::TokenStream;
use proc_macro2::{Delimiter, Ident, TokenTree};
use quote::quote;
use std::collections::HashMap;
use syn::parse::{Parse, ParseStream};
use syn::token::Paren;
use syn::{braced, parenthesized, parse_macro_input, Token};

struct CloneFrom {
    name: Ident,
    fields: HashMap<Ident, proc_macro2::TokenStream>,
}

impl Parse for CloneFrom {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name = input.step(|x| {
            let mut current = *x;
            let mut name: Option<Ident> = None;
            loop {
                if let Some((tree, next)) = current.token_tree() {
                    match tree {
                        TokenTree::Group(g) if g.delimiter() == Delimiter::Brace => {
                            break Ok((name, current));
                        }
                        TokenTree::Ident(ident) => {
                            if ident.to_string() != "pub"
                                && ident.to_string() != "struct"
                                && name.is_none()
                            {
                                name = Some(ident);
                            }
                        }
                        _ => {}
                    }
                    current = next
                } else {
                    return Err(x.error("No opening brace found"));
                }
            }
        })?;
        let name = name.ok_or_else(|| input.error("No name found"))?;

        let input = {
            let fields;
            braced!(fields in input);
            fields
        };

        let mut fields = HashMap::new();

        loop {
            let _ = input.parse::<Token![pub]>();
            if input.peek(Paren) {
                let fields;
                parenthesized!(fields in input);
                let _ = fields.parse::<Token!(crate)>();
            }
            let key: Ident = input.parse()?;

            input.parse::<Token![:]>()?;

            let types = input.step(|x| {
                let mut back = proc_macro2::TokenStream::new();
                let mut current = *x;
                loop {
                    if let Some((tree, next)) = current.token_tree() {
                        match tree {
                            TokenTree::Punct(punct) if punct.as_char() == ',' => {
                                if let Some((tree, next)) = next.token_tree() {
                                    if let TokenTree::Punct(punct) = tree
                                        && punct.as_char() == '>'
                                    {
                                        break Ok((back, next.token_tree().unwrap().1));
                                    }
                                }
                                break Ok((back, next));
                            }
                            _ => {
                                back.extend(proc_macro2::TokenStream::from(tree));
                                current = next;
                            }
                        }
                    } else {
                        return Err(x.error("No available type found"));
                    }
                }
            })?;
            fields.insert(key, types);

            if input.is_empty() {
                break;
            }
        }

        Ok(Self { name, fields })
    }
}

pub fn clone_from(token_stream: TokenStream) -> TokenStream {
    let CloneFrom { name, fields } = parse_macro_input!(token_stream as CloneFrom);

    let fields: HashMap<Ident, proc_macro2::TokenStream> = fields
        .into_iter()
        .filter(|(_, value)| !value.to_string().contains("*"))
        .collect();

    let field_name = fields.keys();

    TokenStream::from(quote! {
        impl crate::CloneFrom<#name> for #name {
            fn clone_fields_from(&mut self, other: &Self) {
                #(
                    self.#field_name = other.#field_name.clone();
                )*
            }
        }
    })
}
