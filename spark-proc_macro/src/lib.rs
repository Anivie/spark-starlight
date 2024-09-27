#![feature(proc_macro_quote)]
#![cfg_attr(debug_assertions, allow(warnings))]

mod keyword;

use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;
use syn::parse::{Parse, ParseBuffer, ParseStream};
use syn::token::Brace;
use syn::{braced, bracketed, parse_macro_input, Token};

struct WrapArgs {
    name: Ident,
    fields: String,

    drop: Option<Ident>,
    drop2: Option<Ident>,
}

impl Parse for  WrapArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name = input.parse::<Ident>()?;
        let fields = if input.peek(Brace) {
            let field: ParseBuffer = {
                let content;
                braced!(content in input);
                content
            };
            field.to_string()
        }else {
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
        }else {
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
        }else {
            None
        };

        if input.peek(Token![;]) {
            input.parse::<Token![;]>()?;
        }

        Ok(WrapArgs {
            name,
            fields,
            drop,
            drop2,
        })
    }
}

#[proc_macro]
pub fn wrap_ffmpeg(token_stream: TokenStream) -> TokenStream {
    let WrapArgs {
        name,
        fields,
        drop,
        drop2,
    } = parse_macro_input!(token_stream as WrapArgs);

    let output = quote! {
        println!("{}", "#name");
    };

    TokenStream::from(output)
}

