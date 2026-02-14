"""
Blog API Router - Serves markdown blog posts from the project directory
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import os
import re

router = APIRouter(prefix="/blog", tags=["blog"])

# Blog posts directory - use absolute path to ensure it works
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_current_dir)))
BLOG_DIR = os.path.join(_project_root, "project", "post-train")


class BlogPost(BaseModel):
    """Blog post metadata"""
    id: str
    title: str
    filename: str
    order: int


class BlogPostContent(BaseModel):
    """Blog post with full content"""
    id: str
    title: str
    filename: str
    content: str


@router.get("/posts", response_model=List[BlogPost])
async def list_posts():
    """List all blog posts"""
    posts = []

    if not os.path.exists(BLOG_DIR):
        return posts

    for filename in os.listdir(BLOG_DIR):
        if filename.endswith('.md') and not filename.startswith('.'):
            # Extract order and title from filename like "00-概述与全景图.md"
            match = re.match(r'^(\d+)-(.+)\.md$', filename)
            if match:
                order = int(match.group(1))
                title = match.group(2)
            else:
                order = 99
                title = filename.replace('.md', '')

            posts.append(BlogPost(
                id=filename.replace('.md', ''),
                title=title,
                filename=filename,
                order=order
            ))

    # Sort by order
    posts.sort(key=lambda x: x.order)
    return posts


@router.get("/posts/{post_id}", response_model=BlogPostContent)
async def get_post(post_id: str):
    """Get a specific blog post content"""
    filename = f"{post_id}.md"
    filepath = os.path.join(BLOG_DIR, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Post not found: {post_id}")

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract title from filename
    match = re.match(r'^(\d+)-(.+)$', post_id)
    if match:
        title = match.group(2)
    else:
        title = post_id

    return BlogPostContent(
        id=post_id,
        title=title,
        filename=filename,
        content=content
    )
