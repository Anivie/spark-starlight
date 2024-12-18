use proc_macro::TokenStream;
use proc_macro2::{Ident, TokenTree};
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseBuffer, ParseStream};
use syn::token::Brace;
use syn::{braced, bracketed, parse_macro_input, Token};
use syn::{parse_str, Result};

struct WrapArgs {
    name: Ident,
    generics: Option<String>,
    fields: String,

    drop: Option<Ident>,
    drop2: Option<Ident>,
}

impl Parse for WrapArgs {
    fn parse(input: ParseStream) -> Result<Self> {
        let name = input.parse::<Ident>()?;

        let generics = if input.peek(Token![<]) {
            input.parse::<Token![<]>()?;
            let back = input.step(|cursor| {
                let mut rest = *cursor;
                let mut back = String::new();
                while let Some((tt, next)) = rest.token_tree() {
                    match &tt {
                        TokenTree::Punct(punct) if punct.as_char() == '>' => {
                            return Ok((back, next));
                        }
                        _ => {
                            back.push_str(&tt.to_string());
                            rest = next;
                        }
                    }
                }
                Err(cursor.error("no `>` was found after this point"))
            })?;
            Some(back)
        } else {
            None
        };

        let fields = if input.peek(Brace) {
            let field: ParseBuffer = {
                let content;
                braced!(content in input);
                content
            };

            let back = field.to_string();
            field.step(|x| {
                let mut current = *x;
                while let Some((_, next)) = current.token_tree() {
                    current = next;
                }
                Ok(((), current))
            })?;

            back.to_string()
        } else {
            String::new()
        };

        let drop2 = if input.peek(crate::keyword::drop) && input.peek2(Token![+]) {
            input.parse::<crate::keyword::drop>()?;
            input.parse::<Token![+]>()?;
            let field: ParseBuffer = {
                let content;
                bracketed!(content in input);
                content
            };
            Some(field.parse::<Ident>()?)
        } else {
            None
        };

        let drop = if input.peek(crate::keyword::drop) {
            input.parse::<crate::keyword::drop>()?;
            let field: ParseBuffer = {
                let content;
                bracketed!(content in input);
                content
            };
            Some(field.parse::<Ident>()?)
        } else {
            None
        };

        if input.peek(Token![;]) {
            input.parse::<Token![;]>()?;
        }

        Ok(WrapArgs {
            name,
            generics,
            fields,
            drop,
            drop2,
        })
    }
}

pub fn wrap_ffmpeg(token_stream: TokenStream) -> TokenStream {
    let WrapArgs {
        name,
        generics,
        fields,
        drop,
        drop2,
    } = parse_macro_input!(token_stream as WrapArgs);
    let raw_name = format_ident!("{}Raw", name);

    let generics: proc_macro2::TokenStream = if generics.is_some() {
        parse_str(&format!("<{}>", generics.unwrap_or_default())).expect("Failed to parse generics")
    } else {
        quote! {}
    };

    let fields: proc_macro2::TokenStream = parse_str(&fields).expect("Failed to parse fields");

    let mut output = quote! {
        use crate::ffi::#name as #raw_name;

        #[derive(Debug)]
        pub struct #name #generics {
            pub(crate) inner: *mut #raw_name,
            #fields
        }

        impl std::ops::Deref for #name {
            type Target = #raw_name;
            fn deref(&self) -> &Self::Target {
                unsafe {
                    &*self.inner
                }
            }
        }

        impl std::ops::DerefMut for #name {
            fn deref_mut(&mut self) -> &mut Self::Target {
                unsafe {
                    &mut *self.inner
                }
            }
        }

        impl #generics crate::CloneFrom<#name> for #name #generics {
            fn clone_fields_from(&mut self, other: &Self) {
                unsafe {
                    (*self.inner).clone_fields_from(&*other.inner);
                }
            }
        }
    };

    if drop.is_some() {
        output.extend(quote! {
            impl std::ops::Drop for #name {
                fn drop(&mut self) {
                    unsafe {
                        crate::ffi::#drop(self.inner);
                    }
                }
            }
        });
    }

    if drop2.is_some() {
        output.extend(quote! {
            impl std::ops::Drop for #name {
                fn drop(&mut self) {
                    unsafe {
                        crate::ffi::#drop2(&mut self.inner as *mut *mut #raw_name);
                    }
                }
            }
        });
    }

    TokenStream::from(output)
}
